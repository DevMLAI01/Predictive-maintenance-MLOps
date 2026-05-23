variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region for all resources"
  default     = "us-central1"
}

variable "bucket_name" {
  type        = string
  description = "GCS bucket name for model artifacts and pipeline data"
}

variable "artifact_registry_name" {
  type        = string
  description = "Artifact Registry repository name for container images"
  default     = "predictive-maintenance"
}

variable "vertex_endpoint_display_name" {
  type        = string
  description = "Display name for the Vertex AI prediction endpoint"
  default     = "rul-predictor-endpoint"
}
