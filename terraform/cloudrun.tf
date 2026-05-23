resource "google_cloud_run_v2_service" "streamlit_dashboard" {
  name     = "rul-dashboard"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_name}/dashboard:latest"

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "VERTEX_ENDPOINT_ID"
        value = google_vertex_ai_endpoint.rul_endpoint.name
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }
}

# Allow unauthenticated invocations so the dashboard is publicly accessible.
resource "google_cloud_run_v2_service_iam_member" "dashboard_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.streamlit_dashboard.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
