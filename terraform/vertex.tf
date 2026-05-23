resource "google_vertex_ai_endpoint" "rul_endpoint" {
  name         = "rul-predictor-endpoint"
  display_name = var.vertex_endpoint_display_name
  location     = var.region

  labels = {
    project = "predictive-maintenance"
  }
}
