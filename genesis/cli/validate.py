"""CLI commands for validation and go-live readiness checks."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
import structlog

from genesis.validation.orchestrator import ValidationOrchestrator
from genesis.validation.decision import DecisionEngine, DeploymentTarget
from genesis.validation.history import ValidationHistory
from genesis.validation.report_generator import ReportGenerator

logger = structlog.get_logger(__name__)


@click.group()
def validate():
    """Validation and go-live readiness commands."""
    pass


@validate.command()
@click.option('--pipeline', default='standard', help='Pipeline to run (quick/standard/comprehensive/go_live)')
@click.option('--save-report', is_flag=True, help='Save report to disk')
@click.option('--format', 'output_format', default='markdown', help='Report format (markdown/json/html/all)')
def all(pipeline: str, save_report: bool, output_format: str):
    """Run all validators in specified pipeline."""
    click.echo(f"🚀 Running validation pipeline: {pipeline}")
    
    async def run():
        orchestrator = ValidationOrchestrator()
        report = await orchestrator.run_pipeline(pipeline)
        
        # Display summary
        if report.ready:
            click.echo(click.style("✅ System is GO for launch!", fg='green', bold=True))
        else:
            click.echo(click.style("❌ System is NO-GO", fg='red', bold=True))
        
        click.echo(f"📊 Overall Score: {report.overall_score:.1f}%")
        click.echo(f"✓ Passed: {sum(1 for r in report.results if r.passed)}/{len(report.results)}")
        
        # Show blocking issues
        if report.blocking_issues:
            click.echo(click.style(f"\n🚨 Blocking Issues ({len(report.blocking_issues)}):", fg='red'))
            for issue in report.blocking_issues[:5]:
                click.echo(f"  - {issue.name}: {issue.message}")
        
        # Save report if requested
        if save_report:
            generator = ReportGenerator()
            
            if output_format == 'all':
                paths = generator.save_all_formats(report)
                click.echo(f"\n📁 Reports saved:")
                for fmt, path in paths.items():
                    click.echo(f"  - {fmt}: {path}")
            else:
                # Save single format
                if output_format == 'markdown':
                    content = generator.generate_markdown(report)
                    ext = 'md'
                elif output_format == 'json':
                    content = generator.generate_json(report)
                    ext = 'json'
                elif output_format == 'html':
                    content = generator.generate_html(report)
                    ext = 'html'
                else:
                    click.echo(f"Unknown format: {output_format}")
                    return
                
                path = Path(f"validation_report.{ext}")
                path.write_text(content)
                click.echo(f"\n📁 Report saved: {path}")
        
        # Save to history
        history = ValidationHistory()
        history_id = history.save_report(report)
        click.echo(f"\n💾 Saved to history (ID: {history_id})")
        
        return report
    
    try:
        report = asyncio.run(run())
        sys.exit(0 if report.ready else 1)
    except Exception as e:
        click.echo(click.style(f"❌ Validation failed: {str(e)}", fg='red'))
        sys.exit(1)


@validate.command()
@click.argument('category', type=click.Choice(['technical', 'security', 'operational', 'business']))
def category(category: str):
    """Run validators for a specific category."""
    click.echo(f"🔍 Running {category} validators...")
    
    async def run():
        orchestrator = ValidationOrchestrator()
        
        # Get validators for category
        validator_names = orchestrator.validator_categories.get(category, [])
        
        if not validator_names:
            click.echo(f"No validators found for category: {category}")
            return
        
        click.echo(f"Running {len(validator_names)} validators: {', '.join(validator_names)}")
        
        # Run validators
        results = []
        for name in validator_names:
            if name in orchestrator.validators:
                validator = orchestrator.validators[name]
                try:
                    result = await orchestrator._run_validator(name, category, validator)
                    results.append(result)
                    
                    status = "✅" if result.passed else "❌"
                    click.echo(f"  {status} {name}: {result.score:.1f}%")
                except Exception as e:
                    click.echo(f"  ❌ {name}: Failed - {str(e)}")
        
        # Summary
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        click.echo(f"\n📊 Category Summary:")
        click.echo(f"  Passed: {passed}/{len(results)}")
        click.echo(f"  Average Score: {avg_score:.1f}%")
        
        return passed == len(results)
    
    try:
        success = asyncio.run(run())
        sys.exit(0 if success else 1)
    except Exception as e:
        click.echo(click.style(f"❌ Validation failed: {str(e)}", fg='red'))
        sys.exit(1)


@validate.command()
@click.option('--limit', default=10, help='Number of entries to show')
@click.option('--pipeline', help='Filter by pipeline name')
@click.option('--days', type=int, help='Days to look back')
def history(limit: int, pipeline: Optional[str], days: Optional[int]):
    """View validation history."""
    history_manager = ValidationHistory()
    
    entries = history_manager.get_history(
        limit=limit,
        pipeline_name=pipeline,
        days_back=days
    )
    
    if not entries:
        click.echo("No validation history found")
        return
    
    # Display as table
    click.echo("\n📜 Validation History")
    click.echo("-" * 80)
    
    for entry in entries:
        ready_icon = "✅" if entry.ready else "❌"
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M")
        
        click.echo(
            f"{entry.id:4} | {timestamp} | {entry.pipeline_name:12} | "
            f"{ready_icon} | {entry.overall_score:5.1f}% | "
            f"{entry.passed_count:2}/{entry.validator_count:2} passed | "
            f"{entry.duration_seconds:6.1f}s"
        )


@validate.command()
@click.option('--pipeline', default='go_live', help='Pipeline to validate')
@click.option('--target', type=click.Choice(['development', 'staging', 'production']), 
              default='production', help='Deployment target')
@click.option('--dry-run/--execute', default=True, help='Dry run or execute deployment')
def go_live(pipeline: str, target: str, dry_run: bool):
    """Check go-live readiness and optionally deploy."""
    click.echo(f"🚀 Checking go-live readiness...")
    click.echo(f"   Pipeline: {pipeline}")
    click.echo(f"   Target: {target}")
    click.echo(f"   Mode: {'Dry Run' if dry_run else 'EXECUTE'}")
    click.echo("")
    
    async def run():
        # Run validation
        orchestrator = ValidationOrchestrator()
        report = await orchestrator.run_pipeline(pipeline)
        
        # Make go/no-go decision
        engine = DecisionEngine()
        deployment_target = DeploymentTarget[target.upper()]
        decision = engine.make_decision(report, deployment_target)
        
        # Display decision
        if decision.ready:
            click.echo(click.style("✅ GO - System is ready for deployment", fg='green', bold=True))
        else:
            click.echo(click.style("❌ NO-GO - System is not ready", fg='red', bold=True))
        
        click.echo(f"📊 Score: {decision.score:.1f}%")
        
        if decision.blocking_issues:
            click.echo(click.style(f"\n🚨 Blocking Issues:", fg='red'))
            for issue in decision.blocking_issues[:5]:
                click.echo(f"  - {issue.name}: {issue.message}")
        
        # Perform safety checks
        click.echo("\n🔒 Safety Checks:")
        safety_checks = engine.perform_safety_checks(decision)
        
        all_checks_passed = True
        for check in safety_checks:
            status = "✅" if check['passed'] else "❌"
            click.echo(f"  {status} {check['name']}: {check['message']}")
            if not check['passed']:
                all_checks_passed = False
        
        # Deployment decision
        if decision.deployment_allowed and all_checks_passed:
            click.echo(click.style("\n✅ Deployment is ALLOWED", fg='green'))
            
            if not dry_run:
                if click.confirm("Do you want to proceed with deployment?"):
                    click.echo("\n🚀 Triggering deployment...")
                    result = engine.trigger_deployment(decision, dry_run=False)
                    
                    if result['success']:
                        click.echo(click.style("✅ Deployment successful!", fg='green'))
                    else:
                        click.echo(click.style(f"❌ Deployment failed: {result.get('error')}", fg='red'))
                else:
                    click.echo("Deployment cancelled by user")
            else:
                click.echo("\n📝 Dry run mode - no deployment triggered")
                result = engine.trigger_deployment(decision, dry_run=True)
                click.echo(f"   {result['message']}")
        else:
            click.echo(click.style("\n❌ Deployment is NOT ALLOWED", fg='red'))
            
            if not decision.deployment_allowed:
                click.echo("   Reason: Validation failed")
            if not all_checks_passed:
                click.echo("   Reason: Safety checks failed")
        
        # Save report
        generator = ReportGenerator()
        paths = generator.save_all_formats(report, decision)
        click.echo(f"\n📁 Reports saved in: {paths['markdown'].parent}")
        
        return decision.deployment_allowed
    
    try:
        success = asyncio.run(run())
        sys.exit(0 if success else 1)
    except Exception as e:
        click.echo(click.style(f"❌ Go-live check failed: {str(e)}", fg='red'))
        sys.exit(1)


@validate.command()
@click.argument('id1', type=int)
@click.argument('id2', type=int)
def compare(id1: int, id2: int):
    """Compare two validation reports."""
    history = ValidationHistory()
    
    try:
        comparison = history.compare_reports(id1, id2)
        
        click.echo(f"\n📊 Comparing Report {id1} vs Report {id2}")
        click.echo("=" * 60)
        
        # Score comparison
        score1 = comparison['report1']['overall_score']
        score2 = comparison['report2']['overall_score']
        score_diff = comparison['score_diff']
        
        if score_diff > 0:
            score_symbol = "📈"
            score_color = "green"
        elif score_diff < 0:
            score_symbol = "📉"
            score_color = "red"
        else:
            score_symbol = "➡️"
            score_color = "yellow"
        
        click.echo(f"{score_symbol} Score: {score1:.1f}% → {score2:.1f}% "
                  f"({score_diff:+.1f}%)", color=score_color)
        
        # Validator changes
        validators = comparison['validators']
        
        if validators['improved']:
            click.echo(click.style("\n✅ Improved:", fg='green'))
            for v in validators['improved'][:5]:
                click.echo(f"  - {v['name']}: {v['old_score']:.1f}% → {v['new_score']:.1f}% "
                          f"(+{v['improvement']:.1f}%)")
        
        if validators['degraded']:
            click.echo(click.style("\n❌ Degraded:", fg='red'))
            for v in validators['degraded'][:5]:
                click.echo(f"  - {v['name']}: {v['old_score']:.1f}% → {v['new_score']:.1f}% "
                          f"(-{v['degradation']:.1f}%)")
        
        if validators['added']:
            click.echo(click.style("\n➕ Added:", fg='cyan'))
            for name in validators['added'][:5]:
                click.echo(f"  - {name}")
        
        if validators['removed']:
            click.echo(click.style("\n➖ Removed:", fg='yellow'))
            for name in validators['removed'][:5]:
                click.echo(f"  - {name}")
        
        # Ready status
        ready1 = comparison['report1']['ready']
        ready2 = comparison['report2']['ready']
        
        if comparison['ready_changed']:
            if ready2:
                click.echo(click.style("\n🎉 System became READY for deployment!", fg='green', bold=True))
            else:
                click.echo(click.style("\n⚠️ System is NO LONGER ready for deployment!", fg='red', bold=True))
        
    except Exception as e:
        click.echo(click.style(f"❌ Failed to compare reports: {str(e)}", fg='red'))
        sys.exit(1)


@validate.command()
@click.argument('pipeline', default='go_live')
@click.option('--days', default=7, help='Number of days to analyze')
def trend(pipeline: str, days: int):
    """Show validation score trend."""
    history = ValidationHistory()
    
    try:
        graph = history.generate_trend_graph(pipeline, days)
        click.echo(graph)
    except Exception as e:
        click.echo(f"Unable to generate trend: {str(e)}")


# Integration with main CLI
@click.group()
def cli():
    """Genesis CLI with validation commands."""
    pass

cli.add_command(validate)

if __name__ == '__main__':
    cli()