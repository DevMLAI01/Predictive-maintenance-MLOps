resource "google_artifact_registry_repository" "containers" {
  repository_id = var.artifact_registry_name
  format        = "DOCKER"
  location      = var.region
  description   = "Container images for predictive maintenance pipeline"
}
