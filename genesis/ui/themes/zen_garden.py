"""Zen Garden theme - Calm color scheme for Genesis trading terminal."""



# Zen Garden Color Palette
# NO RED COLORS - using gray for losses to prevent emotional triggers
ZEN_COLORS = {
    # Primary colors
    "primary": "#4a9d83",      # Sage green - calming primary
    "secondary": "#6b8e7f",    # Muted green - secondary accent
    "background": "#1a1f2e",   # Dark slate - easy on eyes
    "surface": "#242936",      # Slightly lighter surface

    # Text colors
    "text": "#c9d1d9",         # Soft white - main text
    "text-muted": "#8b949e",   # Muted gray - secondary text
    "text-disabled": "#484f58", # Dark gray - disabled text

    # Profit/Loss colors (NO RED)
    "profit": "#58a65c",       # Green - profit indicator
    "profit-bright": "#67d669", # Bright green - strong profit
    "loss": "#6e7681",         # Gray - loss indicator (NOT RED)
    "loss-muted": "#484f58",   # Dark gray - muted loss

    # Status colors
    "success": "#58a65c",      # Green - success messages
    "warning": "#d4a72c",      # Amber - warnings (not red)
    "info": "#4a9d83",         # Sage - info messages
    "neutral": "#8b949e",      # Gray - neutral status

    # Border and accent colors
    "border": "#30363d",       # Subtle borders
    "border-focus": "#4a9d83", # Focused element border
    "accent": "#7aa2a0",       # Soft teal accent

    # Special UI elements
    "panel-bg": "#1f2937",     # Panel background
    "input-bg": "#0d1117",     # Input field background
    "button-bg": "#21262d",    # Button background
    "button-hover": "#30363d", # Button hover state
}


def get_zen_garden_theme() -> dict:
    """Get the complete Zen Garden theme configuration."""
    return {
        "name": "Zen Garden",
        "description": "Calm, non-triggering color scheme for focused trading",
        "colors": ZEN_COLORS,
        "css_variables": {
            # Map to Textual CSS variables
            "$primary": ZEN_COLORS["primary"],
            "$secondary": ZEN_COLORS["secondary"],
            "$background": ZEN_COLORS["background"],
            "$surface": ZEN_COLORS["surface"],
            "$error": ZEN_COLORS["warning"],  # Use warning color instead of red
            "$warning": ZEN_COLORS["warning"],
            "$success": ZEN_COLORS["success"],
            "$primary-background": ZEN_COLORS["panel-bg"],
            "$secondary-background": ZEN_COLORS["surface"],
            "$text": ZEN_COLORS["text"],
            "$text-muted": ZEN_COLORS["text-muted"],
            "$text-disabled": ZEN_COLORS["text-disabled"],
            "$border": ZEN_COLORS["border"],
            "$border-focus": ZEN_COLORS["border-focus"],
            "$accent": ZEN_COLORS["accent"],
        }
    }


def apply_zen_theme_css() -> str:
    """Generate CSS string for the Zen Garden theme."""
    theme = get_zen_garden_theme()
    css_lines = []

    # Generate CSS variable definitions
    css_lines.append(":root {")
    for var, color in theme["css_variables"].items():
        css_lines.append(f"    {var}: {color};")
    css_lines.append("}")

    # Additional theme-specific styles
    css_lines.extend([
        "",
        "/* Zen Garden Theme Styles */",
        "Screen {",
        f"    background: {ZEN_COLORS['background']};",
        "}",
        "",
        "Input {",
        f"    background: {ZEN_COLORS['input-bg']};",
        f"    color: {ZEN_COLORS['text']};",
        f"    border: solid {ZEN_COLORS['border']};",
        "}",
        "",
        "Input:focus {",
        f"    border: solid {ZEN_COLORS['border-focus']};",
        "}",
        "",
        "Static {",
        f"    color: {ZEN_COLORS['text']};",
        "}",
        "",
        "Container {",
        f"    background: {ZEN_COLORS['panel-bg']};",
        f"    border: solid {ZEN_COLORS['border']};",
        "}",
        "",
        "/* Profit/Loss specific colors */",
        ".profit {",
        f"    color: {ZEN_COLORS['profit']};",
        "}",
        "",
        ".loss {",
        f"    color: {ZEN_COLORS['loss']};",
        "}",
        "",
        "/* Status message colors */",
        ".status-success {",
        f"    color: {ZEN_COLORS['success']};",
        "}",
        "",
        ".status-warning {",
        f"    color: {ZEN_COLORS['warning']};",
        "}",
        "",
        ".status-info {",
        f"    color: {ZEN_COLORS['info']};",
        "}",
    ])

    return "\n".join(css_lines)


# Color utility functions
def get_pnl_color(value: float) -> str:
    """Get color for P&L display based on value."""
    if value > 0:
        return ZEN_COLORS["profit"]
    elif value < 0:
        return ZEN_COLORS["loss"]  # Gray, not red
    else:
        return ZEN_COLORS["neutral"]


def get_status_color(status: str) -> str:
    """Get color for status messages."""
    status_map = {
        "success": ZEN_COLORS["success"],
        "warning": ZEN_COLORS["warning"],
        "error": ZEN_COLORS["warning"],  # Use warning instead of red
        "info": ZEN_COLORS["info"],
        "neutral": ZEN_COLORS["neutral"],
    }
    return status_map.get(status.lower(), ZEN_COLORS["text"])
