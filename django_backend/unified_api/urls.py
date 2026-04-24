from django.urls import path

from unified_api import views

urlpatterns = [
    path("health", views.HealthView.as_view()),
    path("estimate", views.EstimateView.as_view()),
    path("estimate-upload", views.EstimateUploadView.as_view()),
    path("ingest-and-value", views.IngestAndValueView.as_view()),
    path("ingest-and-value/batch", views.IngestAndValueBatchView.as_view()),
    path("recent-listings", views.RecentListingsView.as_view()),
    path("listings", views.ListingsView.as_view()),
    path("listings/add-from-valuation", views.AddFromValuationView.as_view()),
    path("listings/<str:listing_id>", views.ListingByIdView.as_view()),
    path("scheduler/start", views.SchedulerStartView.as_view()),
    path("scheduler/stop", views.SchedulerStopView.as_view()),
    path("scheduler/status", views.SchedulerStatusView.as_view()),
    path("scheduler/run-once", views.SchedulerRunOnceView.as_view()),
    path("services", views.ServicesView.as_view()),
]
