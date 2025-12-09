terraform {
  required_providers {
    runpod = {
      source = "decentralized-infrastructure/runpod"
    }
    sops = {
      source = "carlpett/sops"
    }
  }
}

data "sops_file" "secrets" {
  source_file = "${path.module}/../secrets.yaml"
}

provider "runpod" {
  api_key = data.sops_file.secrets.data["RUNPOD_API_KEY"]
}

variable "pod_name" {
  description = "Name for the GPU pod"
  type        = string
  default     = "rna-fm"
}

resource "runpod_pod" "rna_fm" {
  name = var.pod_name

  image_name = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204"

  gpu_type_ids = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
  ]

  gpu_count = 1

  cloud_type        = "SECURE"
  support_public_ip = true

  volume_in_gb         = 50
  container_disk_in_gb = 20

  env = {
    PYTHONUNBUFFERED = "1"
  }

  ports = ["22/tcp"]
}

output "pod_id" {
  description = "The ID of the created pod"
  value       = runpod_pod.rna_fm.id
}

output "datacenter" {
  description = "The actual datacenter where the pod is deployed"
  value       = runpod_pod.rna_fm.actual_data_center
}

output "cost_per_hour" {
  description = "Cost per hour in RunPod credits"
  value       = runpod_pod.rna_fm.cost_per_hr
}
