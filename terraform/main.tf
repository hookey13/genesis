terraform {
  required_version = ">= 1.7.0"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

variable "do_token" {
  description = "DigitalOcean API Token"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "sgp1"
}

resource "digitalocean_droplet" "genesis_prod" {
  image    = "ubuntu-22-04-x64"
  name     = "genesis-prod"
  region   = var.region
  size     = "s-2vcpu-4gb"
  
  ssh_keys = [digitalocean_ssh_key.genesis.fingerprint]
  
  tags = ["genesis", "production", "trading"]
}