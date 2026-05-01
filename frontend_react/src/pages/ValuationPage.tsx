import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  addFromValuation,
  estimate,
  estimateUpload,
} from "../api";
import type { PropertyRequest } from "../types";

const defaultForm: PropertyRequest = {
  property_type: "",
  governorate: "",
  city: "",
  neighborhood: "",
  size_m2: 100,
  bedrooms: 0,
  bathrooms: 0,
  condition: "Good",
  has_pool: false,
  has_garden: false,
  has_parking: false,
  sea_view: false,
  elevator: false,
  description: "",
  uploaded_images_count: 0,
  image_refs: [],
  transaction_type: "sale",
};

const sampleForm: PropertyRequest = {
  property_type: "Appartement",
  governorate: "Tunis",
  city: "La Marsa",
  neighborhood: "Sidi Dhrif",
  size_m2: 128,
  bedrooms: 3,
  bathrooms: 2,
  condition: "Excellent",
  has_pool: false,
  has_garden: true,
  has_parking: true,
  sea_view: true,
  elevator: true,
  description:
    "Bright family apartment with sea exposure, renovated finishes, private parking, and close access to schools and transport.",
  uploaded_images_count: 0,
  image_refs: [],
  transaction_type: "sale",
};

type Dict = Record<string, unknown>;
type XaiTab = "overview" | "drivers" | "evidence" | "risk" | "scenarios" | "nlp";
type ShapViewMode = "absolute" | "signed";
type ConflictWarning = {
  selectedPropertyType: string;
  inferredPropertyType: string;
  message: string;
};

type ComparableView = {
  address: string;
  price: number | null;
  size: number | null;
  similarity: number | null;
  transactionType: string;
  difference: number | null;
};

function mapPropertyTypeLabel(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (normalized === "maison" || normalized === "house") {
    return "House";
  }
  if (normalized === "appartement" || normalized === "apartment" || normalized === "appartment") {
    return "Appartment";
  }
  if (normalized === "terrain" || normalized === "land") {
    return "Land";
  }
  return value;
}

function parseConflictWarning(err: unknown): ConflictWarning | null {
  if (!(err instanceof Error) || !err.message) {
    return null;
  }

  try {
    const payload = JSON.parse(err.message) as Dict;
    const detail = (payload.detail && typeof payload.detail === "object")
      ? (payload.detail as Dict)
      : null;
    if (!detail) {
      return null;
    }

    const code = typeof detail.code === "string" ? detail.code : "";
    if (code !== "property_type_conflict_requires_confirmation") {
      return null;
    }

    return {
      selectedPropertyType: typeof detail.selected_property_type === "string"
        ? mapPropertyTypeLabel(detail.selected_property_type)
        : "your selected type",
      inferredPropertyType: typeof detail.inferred_property_type === "string"
        ? mapPropertyTypeLabel(detail.inferred_property_type)
        : "the inferred type",
      message: typeof detail.message === "string" ? detail.message : "The selected property type conflicts with image inference.",
    };
  } catch {
    return null;
  }
}

function asDictArray(value: unknown): Dict[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is Dict => typeof item === "object" && item !== null);
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asLooseNumber(value: unknown): number | null {
  const strict = asNumber(value);
  if (strict !== null) {
    return strict;
  }
  if (typeof value === "string") {
    const parsed = Number(value.replace(/[^\d.-]/g, ""));
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function formatTnd(value: unknown): string {
  const num = asNumber(value);
  return num === null ? "N/A" : `${Math.round(num).toLocaleString()} TND`;
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export default function ValuationPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [form, setForm] = useState<PropertyRequest>(defaultForm);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conflictWarning, setConflictWarning] = useState<ConflictWarning | null>(null);
  const [uploadImages, setUploadImages] = useState<File[]>([]);
  const [confirmVisualConflict, setConfirmVisualConflict] = useState(false);
  const [valuationResult, setValuationResult] = useState<Dict | null>(null);
  const [activeXaiTab, setActiveXaiTab] = useState<XaiTab>("overview");
  const [driversLimit, setDriversLimit] = useState(10);
  const [driversSignFilter, setDriversSignFilter] = useState<"all" | "positive" | "negative">("all");
  const [selectedDriverFeature, setSelectedDriverFeature] = useState<string | null>(null);
  const [shapLimit, setShapLimit] = useState(10);
  const [shapSignFilter, setShapSignFilter] = useState<"all" | "positive" | "negative">("all");
  const [shapViewMode, setShapViewMode] = useState<ShapViewMode>("absolute");
  const [selectedShapFeature, setSelectedShapFeature] = useState<string | null>(null);
  const [selectedComparableIndex, setSelectedComparableIndex] = useState<number | null>(null);
  const [showRawResult, setShowRawResult] = useState(false);
  const [summaryReport, setSummaryReport] = useState("");

  useEffect(() => {
    const incoming = (location.state as { prefill?: PropertyRequest } | null)?.prefill;
    if (incoming) {
      setForm(incoming);
    }
  }, [location.state]);

  const update = <K extends keyof PropertyRequest>(key: K, value: PropertyRequest[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  // When property type is Land/Terrain, disable and clear residential amenities
  useEffect(() => {
    const p = String(form.property_type || "").trim().toLowerCase();
    const isLand = p === "terrain" || p === "land" || p === "lot";
    if (isLand) {
      setForm((prev) => ({
        ...prev,
        has_pool: false,
        has_garden: false,
        has_parking: false,
        sea_view: false,
        elevator: false,
      }));
    }
  }, [form.property_type]);

  const isLand = String(form.property_type || "").trim().toLowerCase() === "terrain" || String(form.property_type || "").trim().toLowerCase() === "land" || String(form.property_type || "").trim().toLowerCase() === "lot";

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await addFromValuation(form);
      navigate("/listings");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add listing");
    } finally {
      setSubmitting(false);
    }
  };

  const onEstimateOnly = async () => {
    setError(null);
    try {
      const result = await estimate(form);
      setValuationResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Estimate failed");
    }
  };

  const onEstimateUpload = async () => {
    setError(null);
    setConflictWarning(null);
    try {
      const result = await estimateUpload(form, uploadImages, confirmVisualConflict);
      setValuationResult(result);
    } catch (err) {
      const conflict = parseConflictWarning(err);
      if (conflict) {
        setConflictWarning(conflict);
        return;
      }
      setError(err instanceof Error ? err.message : "Upload estimate failed");
    }
  };

  const onContinueWithConflict = async () => {
    setConfirmVisualConflict(true);
    setConflictWarning(null);
    setError(null);
    try {
      const result = await estimateUpload(form, uploadImages, true);
      setValuationResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload estimate failed");
    }
  };

  const onAutoFillSample = () => {
    setForm(sampleForm);
    setError(null);
  };

  const onGenerateSummaryReport = () => {
    if (!valuationResult) {
      return;
    }

    const lines = [
      "EstateMind XAI Summary Report",
      `Generated at: ${new Date().toLocaleString()}`,
      "",
      "Property Inputs",
      `- Type: ${mapPropertyTypeLabel(form.property_type || "Unknown")}`,
      `- Location: ${form.city || "N/A"}, ${form.governorate || "N/A"}`,
      `- Neighborhood: ${form.neighborhood || "N/A"}`,
      `- Size: ${form.size_m2} m2 | Bedrooms: ${form.bedrooms} | Bathrooms: ${form.bathrooms}`,
      `- Condition: ${form.condition}`,
      `- Transaction: ${form.transaction_type}`,
      `- Amenities: ${[
        form.has_pool ? "Swimming pool" : null,
        form.has_garden ? "Garden" : null,
        form.has_parking ? "Parking" : null,
        form.sea_view ? "Sea view" : null,
        form.elevator ? "Elevator" : null,
      ].filter(Boolean).join(", ") || "None"}`,
      "",
      "Valuation",
      `- Estimated Price: ${formatTnd(valuationResult.estimated_price)}`,
      `- Interval: ${formatTnd(valuationResult.lower_bound)} to ${formatTnd(valuationResult.upper_bound)}`,
      `- Confidence: ${confidenceDisplay} (${String(valuationResult.confidence_level ?? "N/A")})`,
      `- Prediction Mode: ${String(valuationResult.prediction_mode ?? "N/A")}`,
      `- Explanation Mode: ${String(valuationResult.explanation_mode ?? "N/A")}`,
      "",
      "XAI Diagnostics",
      `- Feature Impact Rows: ${impacts.length}`,
      `- SHAP Rows: ${shap.length}`,
      `- Scenario Rows: ${scenarioRows.length}`,
      `- Recommendation Rows: ${recommendationRows.length}`,
      `- NLP Sentiment Tokens: ${sentimentTokenRows.length}`,
      `- NLP Quality Tokens: ${qualityTokenRows.length}`,
      `- Comparable Rows: ${comparables.length}`,
      `- Warning Count: ${warnings.length}`,
      `- Uncertainty Reasons: ${uncertaintyReasons.length}`,
      `- Vision Guidance Items: ${visionGuidance.length}`,
      "",
      "Explanation",
      `${String(valuationResult.ai_explanation ?? "N/A")}`,
      "",
      warnings.length > 0 ? `Warnings: ${warnings.join(" | ")}` : "Warnings: none",
      uncertaintyReasons.length > 0 ? `Uncertainty Reasons: ${uncertaintyReasons.join(" | ")}` : "Uncertainty Reasons: none",
    ];

    setSummaryReport(lines.join("\n"));
  };

  const onDownloadSummaryReport = () => {
    if (!summaryReport) {
      return;
    }
    const file = new Blob([summaryReport], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(file);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `xai-summary-report-${Date.now()}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const impacts = asDictArray(valuationResult?.features_impact);
  const shap = asDictArray(valuationResult?.shap);
  const comparables = asDictArray(valuationResult?.comparables);
  const warnings = asStringArray(valuationResult?.warnings);
  const uncertaintyReasons = asStringArray(valuationResult?.uncertainty_reasons);
  const imageAnalysis = asStringArray(valuationResult?.image_analysis);
  const scenarioRows = asDictArray(valuationResult?.scenarios);
  const recommendationRows = asDictArray(valuationResult?.recommendations);
  const sentimentTokenRows = asDictArray(valuationResult?.nlp_sentiment_tokens);
  const qualityTokenRows = asDictArray(valuationResult?.nlp_quality_tokens);
  const locationComparison = (valuationResult?.location_comparison && typeof valuationResult.location_comparison === "object")
    ? (valuationResult.location_comparison as Dict)
    : null;
  const marketContext = (valuationResult?.market_context && typeof valuationResult.market_context === "object")
    ? (valuationResult.market_context as Dict)
    : null;
  const textAnalysis = (valuationResult?.text_analysis && typeof valuationResult.text_analysis === "object")
    ? (valuationResult.text_analysis as Dict)
    : null;
  const modelInfo = (valuationResult?.model_info && typeof valuationResult.model_info === "object")
    ? (valuationResult.model_info as Dict)
    : null;
  const visionGuidance = asDictArray(valuationResult?.vision_guidance);
  const confidenceDisplay = typeof valuationResult?.confidence === "number"
    ? `${valuationResult.confidence}%`
    : "N/A";
  const estimatedPrice = asNumber(valuationResult?.estimated_price) ?? 0;
  const lowerBound = asNumber(valuationResult?.lower_bound) ?? 0;
  const upperBound = asNumber(valuationResult?.upper_bound) ?? 0;
  const intervalRange = Math.max(0, upperBound - lowerBound);
  const intervalSpreadPct = estimatedPrice > 0 ? clampPercent((intervalRange / estimatedPrice) * 100) : 0;
  const confidenceValue = clampPercent(asNumber(valuationResult?.confidence) ?? 0);
  const completeXaiScore = clampPercent(
    (comparables.length > 0 ? 20 : 0)
    + (impacts.length > 0 ? 20 : 0)
    + (shap.length > 0 ? 20 : 0)
    + (typeof valuationResult?.ai_explanation === "string" && valuationResult.ai_explanation.length > 0 ? 20 : 0)
    + (warnings.length === 0 ? 20 : 8)
  );
  const evidenceScore = clampPercent(
    Math.round(
      (comparables.length >= 4 ? 40 : comparables.length * 10)
      + (imageAnalysis.length > 0 ? 30 : 0)
      + (textAnalysis ? 30 : 0)
    )
  );
  const driverRows = useMemo(
    () => impacts
      .map((item) => ({
        feature: String(item.feature ?? "unknown_feature"),
        value: asNumber(item.amount ?? item.value ?? item.impact) ?? 0,
        pct: asNumber(item.pct),
      }))
      .filter((item) => item.feature.trim().length > 0),
    [impacts]
  );
  const filteredDriverRows = useMemo(() => {
    const bySign = driverRows.filter((row) => {
      if (driversSignFilter === "positive") {
        return row.value > 0;
      }
      if (driversSignFilter === "negative") {
        return row.value < 0;
      }
      return true;
    });

    return bySign
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, driversLimit);
  }, [driverRows, driversLimit, driversSignFilter]);
  const filteredDriverMaxAbs = Math.max(1, ...filteredDriverRows.map((row) => Math.abs(row.value)));
  const selectedDriver = filteredDriverRows.find((row) => row.feature === selectedDriverFeature) ?? null;
  const shapContributors = shap.filter((item) => {
    const feature = String(item.feature ?? "").trim().toLowerCase();
    return feature !== "baseline" && feature !== "final";
  });
  const shapDisplayRows = shapContributors.length > 0 ? shapContributors : shap;
  const shapRows = useMemo(
    () => shapDisplayRows
      .map((item) => ({
        feature: String(item.feature ?? "unknown_feature"),
        value: asNumber(item.value ?? item.contribution) ?? 0,
      }))
      .filter((item) => item.feature.trim().length > 0),
    [shapDisplayRows]
  );
  const filteredShapRows = useMemo(() => {
    const bySign = shapRows.filter((row) => {
      if (shapSignFilter === "positive") {
        return row.value > 0;
      }
      if (shapSignFilter === "negative") {
        return row.value < 0;
      }
      return true;
    });

    const sorted = [...bySign].sort((a, b) => {
      if (shapViewMode === "signed") {
        return b.value - a.value;
      }
      return Math.abs(b.value) - Math.abs(a.value);
    });
    return sorted.slice(0, shapLimit);
  }, [shapRows, shapSignFilter, shapViewMode, shapLimit]);
  const filteredShapMaxAbs = Math.max(1, ...filteredShapRows.map((row) => Math.abs(row.value)));
  const selectedShapRow = filteredShapRows.find((row) => row.feature === selectedShapFeature) ?? null;
  const comparableRows = useMemo<ComparableView[]>(
    () => comparables.map((item) => ({
      address: String(item.address ?? "Comparable listing"),
      price: asLooseNumber(item.price ?? item.predicted_price_tnd),
      size: asLooseNumber(item.size),
      similarity: asLooseNumber(item.similarity),
      transactionType: String(item.transaction_type ?? "unknown"),
      difference: asLooseNumber(item.difference),
    })),
    [comparables]
  );
  const selectedComparable = selectedComparableIndex !== null ? comparableRows[selectedComparableIndex] ?? null : null;
  const linkedDrivers = useMemo(() => {
    if (!selectedComparable) {
      return filteredDriverRows.slice(0, 3);
    }
    if (selectedComparable.difference === null) {
      return filteredDriverRows.slice(0, 3);
    }
    if (selectedComparable.difference > 0) {
      return filteredDriverRows.filter((row) => row.value > 0).slice(0, 3);
    }
    if (selectedComparable.difference < 0) {
      return filteredDriverRows.filter((row) => row.value < 0).slice(0, 3);
    }
    return filteredDriverRows.slice(0, 3);
  }, [selectedComparable, filteredDriverRows]);

  const driverChartData = useMemo(
    () => filteredDriverRows.map((row) => ({
      ...row,
      positiveValue: row.value > 0 ? row.value : 0,
      negativeValue: row.value < 0 ? row.value : 0,
    })),
    [filteredDriverRows]
  );
  const shapChartData = useMemo(
    () => filteredShapRows.map((row) => ({
      ...row,
      absValue: Math.abs(row.value),
      positiveSigned: row.value > 0 ? row.value : 0,
      negativeSigned: row.value < 0 ? row.value : 0,
      positiveAbs: row.value > 0 ? Math.abs(row.value) : 0,
      negativeAbs: row.value < 0 ? Math.abs(row.value) : 0,
    })),
    [filteredShapRows]
  );
  const comparableChartData = useMemo(
    () => comparableRows
      .map((item, index) => ({
        ...item,
        index,
        shortLabel: `C${index + 1}`,
        chartPrice: item.price,
        chartSimilarity: item.similarity,
      }))
      .filter((item) => item.chartPrice !== null || item.chartSimilarity !== null),
    [comparableRows]
  );
  const comparableSimilarityMax = useMemo(() => {
    const max = Math.max(1, ...comparableChartData.map((item) => item.chartSimilarity ?? 0));
    return max <= 1 ? 1 : Math.ceil(max);
  }, [comparableChartData]);

  useEffect(() => {
    if (!comparableRows.length) {
      setSelectedComparableIndex(null);
      return;
    }
    if (selectedComparableIndex !== null && selectedComparableIndex >= comparableRows.length) {
      setSelectedComparableIndex(null);
    }
  }, [comparableRows, selectedComparableIndex]);

  return (
    <section className="workspace-panel reveal-panel">
      <div className="panel-header">
        <p className="section-label">VALUATION ENGINE</p>
        <h2 className="panel-title">Decision and Explainability Studio</h2>
        <p className="panel-description">
          Run valuation and inspect explainability outputs, including SHAP contributions, feature impact ranking, comparables, and warning signals.
        </p>
      </div>

      {error && <p className="error">{error}</p>}
      {conflictWarning && (
        <div className="popup-overlay" role="dialog" aria-modal="true" aria-label="Property type conflict warning">
          <div className="popup-warning">
            <h3>Heads up: property type mismatch</h3>
            <p>
              We found a mismatch between the property type you selected and what the uploaded images suggest.
            </p>
            <p className="muted">
              Selected: <strong>{conflictWarning.selectedPropertyType}</strong> | Image inference: <strong>{conflictWarning.inferredPropertyType}</strong>
            </p>
            <p className="muted">
              You can review the property type, or continue anyway if you are sure your selection is correct.
            </p>
            <div className="actions-row">
              <button type="button" onClick={() => setConflictWarning(null)}>Review property type</button>
              <button type="button" onClick={onContinueWithConflict}>Continue anyway</button>
            </div>
          </div>
        </div>
      )}
      <form className="form-grid" onSubmit={onSubmit}>
        <label>
          Property Type
          <select value={form.property_type} onChange={(e) => update("property_type", e.target.value as PropertyRequest["property_type"])}>
            <option value="">Unknown</option>
            <option value="Terrain">Land</option>
            <option value="Maison">House</option>
            <option value="Appartement">Appartment</option>
          </select>
        </label>
        <label>
          Governorate
          <input value={form.governorate} onChange={(e) => update("governorate", e.target.value)} required />
        </label>
        <label>
          City
          <input value={form.city} onChange={(e) => update("city", e.target.value)} required />
        </label>
        <label>
          Neighborhood
          <input value={form.neighborhood} onChange={(e) => update("neighborhood", e.target.value)} />
        </label>
        <label>
          Size (m2)
          <input type="number" min={11} value={form.size_m2} onChange={(e) => update("size_m2", Number(e.target.value))} required />
        </label>
        <label>
          Bedrooms
          <input type="number" min={0} value={form.bedrooms} onChange={(e) => update("bedrooms", Number(e.target.value))} />
        </label>
        <label>
          Bathrooms
          <input type="number" min={0} value={form.bathrooms} onChange={(e) => update("bathrooms", Number(e.target.value))} />
        </label>
        <label>
          Condition
          <select value={form.condition} onChange={(e) => update("condition", e.target.value as PropertyRequest["condition"])}>
            <option value="New">New</option>
            <option value="Excellent">Excellent</option>
            <option value="Good">Good</option>
            <option value="Fair">Fair</option>
            <option value="Needs Renovation">Needs Renovation</option>
          </select>
        </label>
        <label>
          Transaction Type
          <select value={form.transaction_type} onChange={(e) => update("transaction_type", e.target.value as PropertyRequest["transaction_type"])}>
            <option value="sale">Sales</option>
            <option value="rent">Rent</option>
          </select>
        </label>
        <div className="full-row amenities-section">
          <p className="amenities-title">Amenities</p>
          <div className="amenities-grid">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={form.has_pool}
                onChange={(e) => update("has_pool", e.target.checked)}
                disabled={isLand}
                aria-disabled={isLand}
              />
              Has swimming pool
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={form.has_garden}
                onChange={(e) => update("has_garden", e.target.checked)}
                disabled={isLand}
                aria-disabled={isLand}
              />
              Has garden
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={form.has_parking}
                onChange={(e) => update("has_parking", e.target.checked)}
                disabled={isLand}
                aria-disabled={isLand}
              />
              Has parking
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={form.sea_view}
                onChange={(e) => update("sea_view", e.target.checked)}
                disabled={isLand}
                aria-disabled={isLand}
              />
              Sea view
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={form.elevator}
                onChange={(e) => update("elevator", e.target.checked)}
                disabled={isLand}
                aria-disabled={isLand}
              />
              Elevator
            </label>
          </div>
        </div>
        <label className="full-row">
          Description
          <textarea value={form.description} onChange={(e) => update("description", e.target.value)} rows={4} />
        </label>
        <div className="full-row actions-row">
          <button type="submit" disabled={submitting}>{submitting ? "Saving..." : "Run valuation and add"}</button>
          <button type="button" onClick={onEstimateOnly}>Run estimate only</button>
          <button type="button" onClick={onAutoFillSample}>Auto-fill sample data</button>
        </div>

        <label className="full-row">
          Upload images for /estimate-upload
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={(e) => setUploadImages(Array.from(e.target.files ?? []))}
          />
        </label>

        <label className="full-row checkbox-row">
          <input
            type="checkbox"
            checked={confirmVisualConflict}
            onChange={(e) => setConfirmVisualConflict(e.target.checked)}
          />
          Confirm visual property-type conflict for upload estimates
        </label>

        <div className="full-row actions-row">
          <button type="button" onClick={onEstimateUpload}>Run estimate upload</button>
          <span className="muted">{uploadImages.length} image(s) selected</span>
        </div>
      </form>

      {valuationResult && (
        <section className="xai-panel reveal-panel">
          <h3>XAI Results</h3>
          <p className="muted">
            Explanation, confidence, SHAP attribution, comparables, and model diagnostics from the latest valuation call.
          </p>

          <div className="xai-tabs" role="tablist" aria-label="XAI dashboard sections">
            <button type="button" className={`xai-tab ${activeXaiTab === "overview" ? "active" : ""}`} onClick={() => setActiveXaiTab("overview")}>Overview</button>
            <button type="button" className={`xai-tab ${activeXaiTab === "drivers" ? "active" : ""}`} onClick={() => setActiveXaiTab("drivers")}>Drivers</button>
            <button type="button" className={`xai-tab ${activeXaiTab === "evidence" ? "active" : ""}`} onClick={() => setActiveXaiTab("evidence")}>Evidence</button>
            <button type="button" className={`xai-tab ${activeXaiTab === "risk" ? "active" : ""}`} onClick={() => setActiveXaiTab("risk")}>Risk and Warnings</button>
            <button type="button" className={`xai-tab ${activeXaiTab === "scenarios" ? "active" : ""}`} onClick={() => setActiveXaiTab("scenarios")}>Scenarios</button>
            <button type="button" className={`xai-tab ${activeXaiTab === "nlp" ? "active" : ""}`} onClick={() => setActiveXaiTab("nlp")}>NLP Insights</button>
          </div>

          {activeXaiTab === "overview" && <>
          <div className="xai-dashboard-illustration">
            <div className="xai-dashboard-head">
              <h4>XAI Dashboard Summary</h4>
              <span className="xai-dashboard-pill">Illustrated overview</span>
            </div>
            <div className="xai-dashboard-grid">
              <article className="xai-dashboard-card">
                <p>Confidence Signal</p>
                <strong>{confidenceDisplay}</strong>
                <div className="xai-meter"><span style={{ width: `${confidenceValue}%` }} /></div>
              </article>
              <article className="xai-dashboard-card">
                <p>Interval Spread</p>
                <strong>{intervalSpreadPct.toFixed(1)}%</strong>
                <div className="xai-meter caution"><span style={{ width: `${intervalSpreadPct}%` }} /></div>
              </article>
              <article className="xai-dashboard-card">
                <p>XAI Completeness</p>
                <strong>{completeXaiScore.toFixed(0)}%</strong>
                <div className="xai-meter"><span style={{ width: `${completeXaiScore}%` }} /></div>
              </article>
              <article className="xai-dashboard-card">
                <p>Evidence Quality</p>
                <strong>{evidenceScore.toFixed(0)}%</strong>
                <div className="xai-meter"><span style={{ width: `${evidenceScore}%` }} /></div>
              </article>
            </div>
            <div className="xai-dashboard-foot">
              <span>Comparables: {comparables.length}</span>
              <span>Warnings: {warnings.length}</span>
              <span>Vision guidance: {visionGuidance.length}</span>
            </div>
          </div>

          <div className="xai-summary-grid">
            <div className="xai-kpi">
              <span>Estimated Price</span>
              <strong>{Number(valuationResult.estimated_price ?? 0).toLocaleString()} TND</strong>
            </div>
            <div className="xai-kpi">
              <span>Confidence Level</span>
              <strong>{String(valuationResult.confidence_level ?? "N/A")}</strong>
            </div>
            <div className="xai-kpi">
              <span>Confidence</span>
              <strong>{confidenceDisplay}</strong>
            </div>
            <div className="xai-kpi">
              <span>Prediction Mode</span>
              <strong>{String(valuationResult.prediction_mode ?? "N/A")}</strong>
            </div>
            <div className="xai-kpi">
              <span>Explanation Mode</span>
              <strong>{String(valuationResult.explanation_mode ?? "N/A")}</strong>
            </div>
            <div className="xai-kpi">
              <span>Sentiment Mode</span>
              <strong>{String(valuationResult.sentiment_mode ?? "N/A")}</strong>
            </div>
            <div className="xai-kpi">
              <span>CV Mode</span>
              <strong>{String(valuationResult.cv_mode ?? "N/A")}</strong>
            </div>
            <div className="xai-kpi">
              <span>Uncertainty Mode</span>
              <strong>{String(valuationResult.uncertainty_mode ?? "N/A")}</strong>
            </div>
            <div className="xai-kpi">
              <span>Price Interval</span>
              <strong>{formatTnd(valuationResult.lower_bound)} - {formatTnd(valuationResult.upper_bound)}</strong>
            </div>
          </div>

          {typeof valuationResult.ai_explanation === "string" && valuationResult.ai_explanation.length > 0 && (
            <div className="xai-block">
              <h4>AI Explanation</h4>
              <p>{valuationResult.ai_explanation}</p>
            </div>
          )}
          </>}

          {activeXaiTab === "drivers" && <>
          <div className="xai-block">
            <h4>Feature Impact</h4>
            {filteredDriverRows.length === 0 ? (
              <p className="muted">No feature impact available.</p>
            ) : (
              <>
                <div className="xai-drivers-controls">
                  <label>
                    Top drivers
                    <select value={driversLimit} onChange={(e) => setDriversLimit(Number(e.target.value))}>
                      <option value={5}>Top 5</option>
                      <option value={10}>Top 10</option>
                      <option value={20}>Top 20</option>
                    </select>
                  </label>
                  <label>
                    Sign
                    <select value={driversSignFilter} onChange={(e) => setDriversSignFilter(e.target.value as "all" | "positive" | "negative") }>
                      <option value="all">All</option>
                      <option value="positive">Positive only</option>
                      <option value="negative">Negative only</option>
                    </select>
                  </label>
                </div>

                <div className="xai-driver-chart" role="list" aria-label="Interactive feature impact chart">
                  <div className="xai-chart-shell">
                    <ResponsiveContainer width="100%" height={350}>
                      <BarChart
                        data={driverChartData}
                        layout="vertical"
                        margin={{ top: 12, right: 14, left: 2, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis type="number" domain={[-filteredDriverMaxAbs, filteredDriverMaxAbs]} stroke="#9fb2ca" />
                        <YAxis type="category" dataKey="feature" width={156} stroke="#c9d7ea" tick={{ fontSize: 11 }} />
                        <Tooltip
                          cursor={{ fill: "rgba(255,255,255,0.06)" }}
                          formatter={(value) => {
                            const num = asLooseNumber(value);
                            return num === null ? "N/A" : num.toLocaleString();
                          }}
                          labelFormatter={(label) => `Feature: ${String(label)}`}
                        />
                        <Legend />
                        <ReferenceLine x={0} stroke="rgba(255,255,255,0.3)" />
                        <Bar name="Positive impact" dataKey="positiveValue">
                          {driverChartData.map((item) => (
                            <Cell
                              key={`impact-pos-${item.feature}`}
                              fill="rgba(52, 211, 153, 0.85)"
                              stroke={selectedDriverFeature === item.feature ? "#ffe1d3" : "transparent"}
                              strokeWidth={selectedDriverFeature === item.feature ? 2 : 0}
                              style={{ cursor: "pointer" }}
                              onClick={() => setSelectedDriverFeature(item.feature)}
                            />
                          ))}
                        </Bar>
                        <Bar name="Negative impact" dataKey="negativeValue">
                          {driverChartData.map((item) => (
                            <Cell
                              key={`impact-neg-${item.feature}`}
                              fill="rgba(239, 68, 68, 0.85)"
                              stroke={selectedDriverFeature === item.feature ? "#ffe1d3" : "transparent"}
                              strokeWidth={selectedDriverFeature === item.feature ? 2 : 0}
                              style={{ cursor: "pointer" }}
                              onClick={() => setSelectedDriverFeature(item.feature)}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {selectedDriver && (
                  <p className="muted xai-driver-detail">
                    Selected driver: <strong>{selectedDriver.feature}</strong> with contribution <strong>{selectedDriver.value.toLocaleString()}</strong>
                  </p>
                )}
              </>
            )}
          </div>

          <div className="xai-block">
            <h4>SHAP Contributions</h4>
            {filteredShapRows.length === 0 ? (
              <p className="muted">No SHAP contributions available.</p>
            ) : (
              <>
                <div className="xai-drivers-controls">
                  <label>
                    Top SHAP rows
                    <select value={shapLimit} onChange={(e) => setShapLimit(Number(e.target.value))}>
                      <option value={5}>Top 5</option>
                      <option value={10}>Top 10</option>
                      <option value={20}>Top 20</option>
                    </select>
                  </label>
                  <label>
                    Sign
                    <select value={shapSignFilter} onChange={(e) => setShapSignFilter(e.target.value as "all" | "positive" | "negative") }>
                      <option value="all">All</option>
                      <option value="positive">Positive only</option>
                      <option value="negative">Negative only</option>
                    </select>
                  </label>
                  <label>
                    Sort mode
                    <select value={shapViewMode} onChange={(e) => setShapViewMode(e.target.value as ShapViewMode)}>
                      <option value="absolute">Absolute impact</option>
                      <option value="signed">Signed value</option>
                    </select>
                  </label>
                </div>

                <div className="xai-driver-chart" role="list" aria-label="Interactive SHAP chart">
                  <div className="xai-chart-shell">
                    <ResponsiveContainer width="100%" height={350}>
                      <BarChart
                        data={shapChartData}
                        layout="vertical"
                        margin={{ top: 12, right: 14, left: 2, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis
                          type="number"
                          domain={shapViewMode === "signed" ? [-filteredShapMaxAbs, filteredShapMaxAbs] : [0, filteredShapMaxAbs]}
                          stroke="#9fb2ca"
                        />
                        <YAxis type="category" dataKey="feature" width={156} stroke="#c9d7ea" tick={{ fontSize: 11 }} />
                        <Tooltip
                          cursor={{ fill: "rgba(255,255,255,0.06)" }}
                          formatter={(value) => {
                            const num = asLooseNumber(value);
                            return num === null ? "N/A" : num.toFixed(3);
                          }}
                          labelFormatter={(label) => `Feature: ${String(label)}`}
                        />
                        <Legend />
                        {shapViewMode === "signed" && <ReferenceLine x={0} stroke="rgba(255,255,255,0.3)" />}
                        <Bar name="Positive SHAP" dataKey={shapViewMode === "signed" ? "positiveSigned" : "positiveAbs"}>
                          {shapChartData.map((item) => (
                            <Cell
                              key={`shap-pos-${item.feature}`}
                              fill="rgba(52, 211, 153, 0.85)"
                              stroke={selectedShapFeature === item.feature ? "#ffe1d3" : "transparent"}
                              strokeWidth={selectedShapFeature === item.feature ? 2 : 0}
                              style={{ cursor: "pointer" }}
                              onClick={() => setSelectedShapFeature(item.feature)}
                            />
                          ))}
                        </Bar>
                        <Bar name="Negative SHAP" dataKey={shapViewMode === "signed" ? "negativeSigned" : "negativeAbs"}>
                          {shapChartData.map((item) => (
                            <Cell
                              key={`shap-neg-${item.feature}`}
                              fill="rgba(239, 68, 68, 0.85)"
                              stroke={selectedShapFeature === item.feature ? "#ffe1d3" : "transparent"}
                              strokeWidth={selectedShapFeature === item.feature ? 2 : 0}
                              style={{ cursor: "pointer" }}
                              onClick={() => setSelectedShapFeature(item.feature)}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {selectedShapRow && (
                  <p className="muted xai-driver-detail">
                    Selected SHAP feature: <strong>{selectedShapRow.feature}</strong> with value <strong>{selectedShapRow.value.toFixed(3)}</strong>
                  </p>
                )}
              </>
            )}
          </div>
          </>}

          {activeXaiTab === "evidence" && <>
          <div className="xai-block">
            <h4>Comparables ({comparables.length})</h4>
            {comparables.length === 0 ? (
              <p className="muted">No comparables returned.</p>
            ) : (
              <>
                {comparableChartData.length > 0 && (
                  <div className="xai-chart-shell">
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart
                        data={comparableChartData}
                        margin={{ top: 12, right: 18, left: 2, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis dataKey="shortLabel" stroke="#9fb2ca" />
                        <YAxis yAxisId="left" stroke="#c9d7ea" tickFormatter={(value) => `${Math.round(Number(value) / 1000)}k`} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, comparableSimilarityMax]} stroke="#f8c3a7" />
                        <Tooltip
                          formatter={(value, name) => {
                            const displayName = String(name);
                            const numericValue = asLooseNumber(value);
                            if (displayName === "Price") {
                              if (numericValue === null) {
                                return ["N/A", displayName];
                              }
                              return [formatTnd(numericValue), displayName];
                            }
                            if (numericValue === null) {
                              return ["N/A", displayName];
                            }
                            return [numericValue.toFixed(3), displayName];
                          }}
                          labelFormatter={(label) => {
                            const row = comparableChartData.find((item) => item.shortLabel === label);
                            return row ? `${label}: ${row.address}` : String(label);
                          }}
                        />
                        <Legend />
                        {estimatedPrice > 0 && <ReferenceLine yAxisId="left" y={estimatedPrice} stroke="rgba(255, 138, 107, 0.85)" label="Estimate" />}
                        <Bar yAxisId="left" dataKey="chartPrice" name="Price">
                          {comparableChartData.map((item) => (
                            <Cell
                              key={`comp-bar-${item.index}`}
                              fill={selectedComparableIndex === item.index ? "rgba(255, 138, 107, 0.95)" : "rgba(80, 174, 255, 0.78)"}
                              stroke={selectedComparableIndex === item.index ? "#ffe1d3" : "transparent"}
                              strokeWidth={selectedComparableIndex === item.index ? 2 : 0}
                              style={{ cursor: "pointer" }}
                              onClick={() => setSelectedComparableIndex(item.index)}
                            />
                          ))}
                        </Bar>
                        <Line yAxisId="right" type="monotone" dataKey="chartSimilarity" name="Similarity" stroke="#f5bf63" dot={{ r: 3 }} activeDot={{ r: 5 }} />
                        <Brush dataKey="shortLabel" height={18} stroke="rgba(255,255,255,0.3)" travellerWidth={8} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                <div className="xai-comparables-grid">
                  {comparableRows.map((item, index) => (
                    <button
                      type="button"
                      className={`xai-comparable-card ${selectedComparableIndex === index ? "active" : ""}`}
                      key={`comp-${index}`}
                      onClick={() => setSelectedComparableIndex(index)}
                    >
                      <h5>{item.address}</h5>
                      <p>
                        Price: <strong>{formatTnd(item.price)}</strong>
                      </p>
                      <p>Size: {item.size === null ? "N/A" : `${item.size} m2`}</p>
                      <p>Similarity: {item.similarity === null ? "N/A" : item.similarity}</p>
                      <p>Transaction: {item.transactionType}</p>
                      <p>Difference: {item.difference === null ? "N/A" : item.difference}</p>
                    </button>
                  ))}
                </div>
              </>
            )}

            {selectedComparable && (
              <div className="xai-comparable-link-panel">
                <h5>Linked Driver Insight</h5>
                <p className="muted">
                  Selected comparable <strong>{selectedComparable.address}</strong> with difference {selectedComparable.difference === null ? "N/A" : selectedComparable.difference}.
                </p>
                {linkedDrivers.length > 0 ? (
                  <ul className="xai-warning-list">
                    {linkedDrivers.map((row) => (
                      <li key={`linked-driver-${row.feature}`}>
                        <button
                          type="button"
                          className="inline-link"
                          onClick={() => {
                            setActiveXaiTab("drivers");
                            setSelectedDriverFeature(row.feature);
                          }}
                        >
                          Open driver: {row.feature} ({row.value.toLocaleString()})
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">No linked drivers available for this comparable.</p>
                )}
              </div>
            )}
          </div>

          {marketContext && (
            <div className="xai-block">
              <h4>Market Context</h4>
              <div className="xai-context-grid">
                <div className="xai-context-item"><span>City</span><strong>{String(marketContext.city ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Market Avg / m2</span><strong>{String(marketContext.avg_m2 ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Property / m2</span><strong>{String(marketContext.property_m2 ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Delta</span><strong>{String(marketContext.delta_pct ?? "N/A")}%</strong></div>
                <div className="xai-context-item"><span>Trend</span><strong>{String(marketContext.trend ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Demand</span><strong>{String(marketContext.demand ?? "N/A")}</strong></div>
              </div>
              {typeof marketContext.trend_reason === "string" && marketContext.trend_reason.length > 0 && (
                <p className="muted">{marketContext.trend_reason}</p>
              )}
            </div>
          )}

          {textAnalysis && (
            <div className="xai-block">
              <h4>Text Analysis Signals</h4>
              <div className="xai-context-grid">
                <div className="xai-context-item"><span>Description Quality</span><strong>{String(textAnalysis.description_quality ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Description Sentiment</span><strong>{String(textAnalysis.description_sentiment_label ?? textAnalysis.description_sentiment ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Location Sentiment</span><strong>{String(textAnalysis.location_sentiment_label ?? textAnalysis.location_sentiment ?? "N/A")}</strong></div>
                <div className="xai-context-item"><span>Marketing Effectiveness</span><strong>{String(textAnalysis.marketing_effectiveness ?? "N/A")}</strong></div>
              </div>
            </div>
          )}

          {imageAnalysis.length > 0 && (
            <div className="xai-block">
              <h4>Image Analysis Summary</h4>
              <ul className="xai-warning-list">
                {imageAnalysis.map((item, index) => (
                  <li key={`image-analysis-${index}`}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {visionGuidance.length > 0 && (
            <div className="xai-block">
              <h4>Vision Guidance</h4>
              <p className="muted">{visionGuidance.length} guidance item(s) returned by the visual analysis pipeline.</p>
              <ul className="xai-warning-list">
                {visionGuidance.map((item, index) => {
                  const primary = typeof item.message === "string"
                    ? item.message
                    : typeof item.label === "string"
                      ? item.label
                      : typeof item.title === "string"
                        ? item.title
                        : typeof item.code === "string"
                          ? item.code
                          : null;
                  const details = typeof item.details === "string"
                    ? item.details
                    : typeof item.description === "string"
                      ? item.description
                      : null;

                  return (
                    <li key={`vision-guidance-${index}`}>
                      {primary ?? JSON.stringify(item)}
                      {details ? `: ${details}` : ""}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
          </>}

          {activeXaiTab === "scenarios" && <>
          <div className="xai-block">
            <h4>Scenario Simulator</h4>
            <p className="muted">Fast what-if recommendations ranked by estimated price uplift.</p>
            {scenarioRows.length === 0 ? (
              <p className="muted">No scenario simulations were returned by the backend.</p>
            ) : (
              <div className="xai-raw-grid">
                {scenarioRows.map((row, index) => {
                  const delta = asLooseNumber(row.price_delta ?? row.predicted_impact_tnd) ?? 0;
                  const predicted = asLooseNumber(row.predicted_price) ?? 0;
                  const pct = asLooseNumber(row.delta_percentage ?? row.predicted_impact_pct) ?? 0;
                  const title = String(row.scenario_name ?? row.title ?? `Scenario ${index + 1}`);
                  const description = String(row.scenario_description ?? row.description ?? "");
                  const why = String(row.why ?? row.justification ?? "");
                  return (
                    <article className="xai-viz-item" key={`scenario-${index}`}>
                      <div className="xai-viz-head">
                        <span>{title}</span>
                        <strong>{delta >= 0 ? "+" : ""}{delta.toLocaleString()} TND ({pct >= 0 ? "+" : ""}{pct.toFixed(2)}%)</strong>
                      </div>
                      <div className="xai-summary-grid">
                        <div className="xai-kpi">
                          <span>Before</span>
                          <strong>{estimatedPrice.toLocaleString()} TND</strong>
                        </div>
                        <div className="xai-kpi">
                          <span>After</span>
                          <strong>{predicted.toLocaleString()} TND</strong>
                        </div>
                        <div className="xai-kpi">
                          <span>Confidence</span>
                          <strong>{String(row.confidence ?? "N/A")}</strong>
                        </div>
                      </div>
                      <p className="muted">{description}</p>
                      {why && <p className="muted">Why: {why}</p>}
                      <p className="muted">Changes: {JSON.stringify(row.modified_features ?? {}, null, 0)}</p>
                    </article>
                  );
                })}
              </div>
            )}
          </div>

          <div className="xai-block">
            <h4>Smart Recommendations</h4>
            {recommendationRows.length === 0 ? (
              <p className="muted">No recommendations were returned.</p>
            ) : (
              <ul className="xai-list">
                {recommendationRows.map((row, index) => {
                  const title = String(row.title ?? row.scenario_name ?? `Recommendation ${index + 1}`);
                  const impactTnd = asLooseNumber(row.predicted_impact_tnd ?? row.price_delta) ?? 0;
                  const impactPct = asLooseNumber(row.predicted_impact_pct ?? row.delta_percentage) ?? 0;
                  return (
                    <li key={`recommendation-${index}`}>
                      <div>
                        <strong>{title}</strong>
                        <p className="muted" style={{ margin: "0.25rem 0 0" }}>{String(row.justification ?? row.description ?? "")}</p>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <strong>{impactTnd >= 0 ? "+" : ""}{impactTnd.toLocaleString()} TND</strong>
                        <p className="muted" style={{ margin: 0 }}>{impactPct >= 0 ? "+" : ""}{impactPct.toFixed(2)}%</p>
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          </>}

          {activeXaiTab === "nlp" && <>
          <div className="xai-block">
            <h4>Word-Level Sentiment Contributions</h4>
            {sentimentTokenRows.length === 0 ? (
              <p className="muted">No sentiment tokens were returned.</p>
            ) : (
              <div className="xai-viz-list">
                {sentimentTokenRows.map((row, index) => {
                  const contribution = asLooseNumber(row.contribution) ?? 0;
                  const magnitude = asLooseNumber(row.magnitude) ?? Math.abs(contribution);
                  return (
                    <div className="xai-viz-item" key={`sentiment-token-${index}`} title={String(row.explanation ?? "")}>
                      <div className="xai-viz-head">
                        <span>{String(row.token ?? `token-${index}`)}</span>
                        <strong>{contribution >= 0 ? "+" : ""}{contribution.toFixed(2)}</strong>
                      </div>
                      <div className="xai-viz-track">
                        <span className={`xai-viz-fill ${contribution >= 0 ? "positive" : "negative"}`} style={{ width: `${Math.max(12, Math.min(100, magnitude * 100))}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="xai-block">
            <h4>Word-Level Quality and Marketing Signals</h4>
            {qualityTokenRows.length === 0 ? (
              <p className="muted">No quality tokens were returned.</p>
            ) : (
              <div className="xai-viz-list">
                {qualityTokenRows.map((row, index) => {
                  const contribution = asLooseNumber(row.contribution) ?? 0;
                  const magnitude = asLooseNumber(row.magnitude) ?? Math.abs(contribution);
                  return (
                    <div className="xai-viz-item" key={`quality-token-${index}`} title={String(row.explanation ?? "")}>
                      <div className="xai-viz-head">
                        <span>{String(row.token ?? `token-${index}`)} <span className="muted">({String(row.aspect ?? "quality")})</span></span>
                        <strong>{contribution >= 0 ? "+" : ""}{contribution.toFixed(2)}</strong>
                      </div>
                      <div className="xai-viz-track">
                        <span className={`xai-viz-fill ${contribution >= 0 ? "positive" : "negative"}`} style={{ width: `${Math.max(12, Math.min(100, magnitude * 100))}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="xai-block">
            <h4>Location Sentiment Comparison</h4>
            {locationComparison ? (
              <>
                <div className="xai-summary-grid">
                  <div className="xai-kpi">
                    <span>Location Sentiment</span>
                    <strong>{asLooseNumber(locationComparison.location_sentiment)?.toFixed(2) ?? "N/A"}</strong>
                  </div>
                  <div className="xai-kpi">
                    <span>Percentile</span>
                    <strong>{asLooseNumber(locationComparison.location_percentile)?.toFixed(1) ?? "N/A"}%</strong>
                  </div>
                  <div className="xai-kpi">
                    <span>Benchmark</span>
                    <strong>{String(locationComparison.location_sentiment_label ?? "neutral")}</strong>
                  </div>
                </div>
                <p className="muted" style={{ marginTop: "0.6rem" }}>{String(locationComparison.benchmark_message ?? "")}</p>
                {Array.isArray(locationComparison.similar_locations) && locationComparison.similar_locations.length > 0 && (
                  <div className="xai-comparables-grid">
                    {(locationComparison.similar_locations as string[]).map((item, index) => (
                      <article className="xai-comparable-card active" key={`location-${index}`}>
                        <h5>{item}</h5>
                        <p>Comparable benchmark location</p>
                      </article>
                    ))}
                  </div>
                )}
              </>
            ) : <p className="muted">No location comparison data returned.</p>}
          </div>
          </>}

          {activeXaiTab === "risk" && <>
          <div className="xai-block">
            <h4>Risk Summary</h4>
            <div className="xai-context-grid">
              <div className="xai-context-item"><span>Warnings</span><strong>{warnings.length}</strong></div>
              <div className="xai-context-item"><span>Uncertainty reasons</span><strong>{uncertaintyReasons.length}</strong></div>
              <div className="xai-context-item"><span>Interval spread</span><strong>{intervalSpreadPct.toFixed(1)}%</strong></div>
              <div className="xai-context-item"><span>Confidence</span><strong>{confidenceDisplay}</strong></div>
            </div>
          </div>

          {typeof valuationResult.vision_requires_confirmation === "boolean" && (
            <div className="xai-block">
              <h4>Vision Confirmation</h4>
              <p className="muted">
                Requires confirmation: {valuationResult.vision_requires_confirmation ? "Yes" : "No"}
              </p>
            </div>
          )}

          {uncertaintyReasons.length > 0 && (
            <div className="xai-block">
              <h4>Uncertainty Reasons</h4>
              <ul className="xai-warning-list">
                {uncertaintyReasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          )}

          {warnings.length > 0 && (
            <div className="xai-block">
              <h4>Warnings</h4>
              <ul className="xai-warning-list">
                {warnings.map((warn) => (
                  <li key={warn}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {modelInfo && (
            <div className="xai-block">
              <h4>Model Info</h4>
              <p className="muted">Model diagnostics available. Use Show Raw JSON to inspect detailed fields.</p>
            </div>
          )}
          </>}

          <div className="actions-row">
            <button type="button" onClick={onGenerateSummaryReport}>Generate summary report</button>
            <button type="button" onClick={onDownloadSummaryReport} disabled={!summaryReport}>Download summary report</button>
            <button type="button" onClick={() => setShowRawResult((prev) => !prev)}>
              {showRawResult ? "Hide raw JSON" : "Show raw JSON"}
            </button>
          </div>

          {summaryReport && (
            <label className="service-input xai-summary-report">
              Summary report
              <textarea value={summaryReport} readOnly rows={14} />
            </label>
          )}

          {showRawResult && (
            <div className="xai-raw-grid">
              {visionGuidance.length > 0 && (
                <label className="service-input">
                  Vision guidance JSON
                  <textarea value={JSON.stringify(visionGuidance, null, 2)} readOnly rows={8} />
                </label>
              )}
              {modelInfo && (
                <label className="service-input">
                  Model info JSON
                  <textarea value={JSON.stringify(modelInfo, null, 2)} readOnly rows={8} />
                </label>
              )}
              <label className="service-input">
                Full valuation payload
                <textarea value={JSON.stringify(valuationResult, null, 2)} readOnly rows={14} />
              </label>
            </div>
          )}
        </section>
      )}
    </section>
  );
}
