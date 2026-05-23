resource "google_storage_bucket" "artifacts" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Grants the default Compute Engine SA (used by Vertex AI Pipelines and Endpoints)
# objectAdmin on the artifacts bucket.
resource "google_storage_bucket_iam_member" "vertex_sa_access" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}
