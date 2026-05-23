output "bucket_uri" {
  description = "GCS bucket URI for model artifacts"
  value       = "gs://${google_storage_bucket.artifacts.name}"
}

output "artifact_registry_url" {
  description = "Artifact Registry base URL for container images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "vertex_endpoint_id" {
  description = "Vertex AI endpoint resource name"
  value       = google_vertex_ai_endpoint.rul_endpoint.name
}

output "dashboard_url" {
  description = "Streamlit dashboard URL on Cloud Run"
  value       = google_cloud_run_v2_service.streamlit_dashboard.uri
}
