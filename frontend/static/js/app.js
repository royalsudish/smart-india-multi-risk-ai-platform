/**
 * Smart Multi-Risk AI Platform - Phase D Geospatial Command Center Controller
 * Unified Pan-India Geospatial Dashboard with Leaflet.js Interactive India Map
 */

// =============================================================================
// 1. STATE & MAP CONFIGURATION
// =============================================================================

let isAnalyzing = false;
let currentRequestId = 0;
let loadingInterval = null;

// Selected Location State (Internal exact coordinates, no rounding)
let selectedLocation = {
  lat: 16.3067,
  lon: 80.4365,
  name: "Guntur, Andhra Pradesh, India",
  isAreaLevel: false
};

// Base Map Configuration (Isolated for easy production provider swapping)
const BASEMAP_CONFIG = {
  url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a>',
  maxZoom: 19,
  subdomains: "abcd",
  fallbackUrl: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  fallbackAttribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors'
};

// Centralized Map State Architecture
const mapState = {
  map: null,
  selectedMarker: null,
  drainageLayer: null,
  drainageBufferCircle: null,
  terrainLayer: null,
  disasterLayer: null,
  fireLayer: null,
  rainfallLayer: null,
  satelliteLayer: null,
  sentinel2Layer: null,
  // Pan-India Hierarchical Spatial Layers
  alertsLayer: null,
  wlGridLayer: null,
  drainGridLayer: null,
  hotspotGridLayer: null,
  stateLayer: null,
  districtLayer: null,
  currentView: "GRID", // "INDIA", "STATE", "DISTRICT", "GRID"
  gridGeoJSONData: null,
  indiaGeoJSONData: null,
  stateGeoJSONData: null,
  // Layer visibility toggles
  activeLayers: {
    wlGrid: true,
    drainGrid: true,
    hotspots: true,
    waterlogging: true,
    drainage: true,
    terrain: false,
    disaster: true,
    fireHotspots: true,
    activeAlerts: true,
    rainfall: true,
    satellite: true,
    sentinel2: true
  },
  // Cache of latest data for layer re-rendering on toggle
  latestData: {
    core: null,
    drainage: null,
    terrain: null,
    rainfall: null,
    satellite: null,
    sentinel2: null,
    gridRisk: null,
    indiaRisk: null,
    stateRisk: null
  }
};

// Cycling Loading Messages (UI feedback only)
const LOADING_STAGES = [
  "Fetching live weather & atmospheric data...",
  "Checking live disaster feeds (Flood, USGS, GDACS)...",
  "Analyzing 14-day agricultural rolling metrics...",
  "Checking waterlogging risk & 7-day rainfall windows...",
  "Running multi-model machine learning inference...",
  "Synthesizing authoritative overall risk & advisory..."
];

// DOM Elements Cache
const dom = {
  latInput: document.getElementById("latInput"),
  lonInput: document.getElementById("lonInput"),
  cropSelect: document.getElementById("cropSelect"),
  cropStageSelect: document.getElementById("cropStageSelect"),
  placeSearchInput: document.getElementById("placeSearchInput"),
  placeSearchResults: document.getElementById("placeSearchResults"),
  searchSpinner: document.getElementById("searchSpinner"),
  searchLocationBtn: document.getElementById("searchLocationBtn"),
  useLocationBtn: document.getElementById("useLocationBtn"),
  gpsBtnText: document.getElementById("gpsBtnText"),
  selectedLocationCard: document.getElementById("selectedLocationCard"),
  selectedPlaceName: document.getElementById("selectedPlaceName"),
  pointAnalysisDisclaimer: document.getElementById("pointAnalysisDisclaimer"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  btnSpinner: document.getElementById("btnSpinner"),
  btnText: document.getElementById("btnText"),
  validationMsg: document.getElementById("validationMsg"),
  loadingIndicator: document.getElementById("loadingIndicator"),
  loadingStageText: document.getElementById("loadingStageText"),

  // Map Elements & Hierarchical Navigation
  mapContainer: document.getElementById("mapContainer"),
  mapResetBtn: document.getElementById("mapResetBtn"),
  legendContent: document.getElementById("legendContent"),
  viewBtnIndia: document.getElementById("viewBtnIndia"),
  viewBtnState: document.getElementById("viewBtnState"),
  viewBtnDistrict: document.getElementById("viewBtnDistrict"),
  viewBtnGrid: document.getElementById("viewBtnGrid"),
  navStateSelect: document.getElementById("navStateSelect"),
  navDistrictSelect: document.getElementById("navDistrictSelect"),
  navGridResolution: document.getElementById("navGridResolution"),
  btnGenerateGrid: document.getElementById("btnGenerateGrid"),
  layerToggleWlGrid: document.getElementById("layerToggleWlGrid"),
  layerToggleDrainGrid: document.getElementById("layerToggleDrainGrid"),
  layerToggleHotspots: document.getElementById("layerToggleHotspots"),
  layerToggleWaterlogging: document.getElementById("layerToggleWaterlogging"),
  layerToggleDrainage: document.getElementById("layerToggleDrainage"),
  layerToggleTerrain: document.getElementById("layerToggleTerrain"),
  layerToggleDisaster: document.getElementById("layerToggleDisaster"),
  layerItemFireHotspots: document.getElementById("layerItemFireHotspots"),
  layerToggleFireHotspots: document.getElementById("layerToggleFireHotspots"),
  layerBadgeFireHotspots: document.getElementById("layerBadgeFireHotspots"),
  layerItemActiveAlerts: document.getElementById("layerItemActiveAlerts"),
  layerToggleActiveAlerts: document.getElementById("layerToggleActiveAlerts"),
  layerBadgeActiveAlerts: document.getElementById("layerBadgeActiveAlerts"),

  // Operational Alerts & Lifecycle Elements
  activeAlertsContainer: document.getElementById("activeAlertsContainer"),
  activeAlertsList: document.getElementById("activeAlertsList"),
  noActiveAlertsBox: document.getElementById("noActiveAlertsBox"),
  activeAlertsCountBadge: document.getElementById("activeAlertsCountBadge"),
  alertHistoryTimeline: document.getElementById("alertHistoryTimeline"),
  alertHistoryCount: document.getElementById("alertHistoryCount"),

  // Alerts Bar Elements
  overallRiskBadge: document.getElementById("overallRiskBadge"),
  alertBarWaterlogging: document.getElementById("alertBarWaterlogging"),
  alertBarOutlook: document.getElementById("alertBarOutlook"),
  alertBarDisaster: document.getElementById("alertBarDisaster"),
  alertBarAgri: document.getElementById("alertBarAgri"),
  summaryModelsCount: document.getElementById("summaryModelsCount"),
  summaryTotalTime: document.getElementById("summaryTotalTime"),
  summaryTimestamp: document.getElementById("summaryTimestamp"),
  summaryLocation: document.getElementById("summaryLocation"),
  overallAdvisoryBox: document.getElementById("overallAdvisoryBox"),
  overallAdvisoryText: document.getElementById("overallAdvisoryText"),

  // Weather Panel
  weatherStatusBadge: document.getElementById("weatherStatusBadge"),
  weatherErrorBox: document.getElementById("weatherErrorBox"),
  weatherContent: document.getElementById("weatherContent"),
  wTemp: document.getElementById("wTemp"),
  wApparentTemp: document.getElementById("wApparentTemp"),
  wHumidity: document.getElementById("wHumidity"),
  wDewPoint: document.getElementById("wDewPoint"),
  wPrecip: document.getElementById("wPrecip"),
  wPressure: document.getElementById("wPressure"),
  wWindSpeed: document.getElementById("wWindSpeed"),
  wWindGusts: document.getElementById("wWindGusts"),
  wCloudCover: document.getElementById("wCloudCover"),
  wSoilMoisture: document.getElementById("wSoilMoisture"),
  wSoilTemp: document.getElementById("wSoilTemp"),
  wET0: document.getElementById("wET0"),
  wRain: document.getElementById("wRain"),
  wWindDirection: document.getElementById("wWindDirection"),
  wVPD: document.getElementById("wVPD"),
  wShortwaveRad: document.getElementById("wShortwaveRad"),
  wDNI: document.getElementById("wDNI"),
  wDiffuseRad: document.getElementById("wDiffuseRad"),

  // Disaster Panel
  disasterStatusBadge: document.getElementById("disasterStatusBadge"),
  disasterQualityBadge: document.getElementById("disasterQualityBadge"),
  disasterErrorBox: document.getElementById("disasterErrorBox"),
  disasterContent: document.getElementById("disasterContent"),
  dVerifiedBox: document.getElementById("dVerifiedBox"),
  dVerifiedStatusBadge: document.getElementById("dVerifiedStatusBadge"),
  dVerifiedHazardBadge: document.getElementById("dVerifiedHazardBadge"),
  dVerifiedSeverity: document.getElementById("dVerifiedSeverity"),
  dMultiHazardsContainer: document.getElementById("dMultiHazardsContainer"),
  dOfficialAlertsContainer: document.getElementById("dOfficialAlertsContainer"),
  dVerifiedReason: document.getElementById("dVerifiedReason"),
  cardWildfireMonitor: document.getElementById("cardWildfireMonitor"),
  badgeWildfireStatus: document.getElementById("badgeWildfireStatus"),
  valFireNearest: document.getElementById("valFireNearest"),
  valFireCounts: document.getElementById("valFireCounts"),
  valFireMaxFrp: document.getElementById("valFireMaxFrp"),
  descWildfireReason: document.getElementById("descWildfireReason"),
  cardAvalancheMonitor: document.getElementById("cardAvalancheMonitor"),
  badgeAvalancheStatus: document.getElementById("badgeAvalancheStatus"),
  valAvalancheElev: document.getElementById("valAvalancheElev"),
  valAvalancheSlope: document.getElementById("valAvalancheSlope"),
  valAvalancheTemp: document.getElementById("valAvalancheTemp"),
  descAvalancheReason: document.getElementById("descAvalancheReason"),
  cardFlashFloodMonitor: document.getElementById("cardFlashFloodMonitor"),
  badgeFlashFloodStatus: document.getElementById("badgeFlashFloodStatus"),
  valFlashFloodRain: document.getElementById("valFlashFloodRain"),
  valFlashFloodProbs: document.getElementById("valFlashFloodProbs"),
  valFlashFloodSar: document.getElementById("valFlashFloodSar"),
  descFlashFloodReason: document.getElementById("descFlashFloodReason"),
  dClass: document.getElementById("dClass"),
  dConfidence: document.getElementById("dConfidence"),
  disasterProbabilitiesBox: document.getElementById("disasterProbabilitiesBox"),
  dProbList: document.getElementById("dProbList"),
  dNearestQuakeDist: document.getElementById("dNearestQuakeDist"),
  dNearestQuakeAge: document.getElementById("dNearestQuakeAge"),
  dMaxQuakeMag: document.getElementById("dMaxQuakeMag"),
  dSeismicStatus: document.getElementById("dSeismicStatus"),
  dRiverDischarge: document.getElementById("dRiverDischarge"),
  dQuakeDepth: document.getElementById("dQuakeDepth"),
  dTsunamiFlag: document.getElementById("dTsunamiFlag"),
  dCycloneFlag: document.getElementById("dCycloneFlag"),
  dVolcanoFlag: document.getElementById("dVolcanoFlag"),
  dQuakes24h: document.getElementById("dQuakes24h"),
  dQuakes7d: document.getElementById("dQuakes7d"),
  dQuakes30d: document.getElementById("dQuakes30d"),
  dQuakes90d: document.getElementById("dQuakes90d"),
  dPeakDischarge7d: document.getElementById("dPeakDischarge7d"),
  dDischargeTrend: document.getElementById("dDischargeTrend"),
  dRain7d: document.getElementById("dRain7d"),
  dDroughtAnomaly: document.getElementById("dDroughtAnomaly"),
  dShear: document.getElementById("dShear"),
  dSaturation: document.getElementById("dSaturation"),
  dThermal: document.getElementById("dThermal"),

  // Agriculture Panel
  agStatusBadge: document.getElementById("agStatusBadge"),
  agErrorBox: document.getElementById("agErrorBox"),
  agContent: document.getElementById("agContent"),
  agCropDisplay: document.getElementById("agCropDisplay"),
  agStageDisplay: document.getElementById("agStageDisplay"),
  agTargetWeatherRisk: document.getElementById("agTargetWeatherRisk"),
  agTargetIrrigation: document.getElementById("agTargetIrrigation"),
  agTargetHeatStress: document.getElementById("agTargetHeatStress"),
  agTargetWaterlogging: document.getElementById("agTargetWaterlogging"),
  agTargetSpraying: document.getElementById("agTargetSpraying"),
  agProbabilitiesBox: document.getElementById("agProbabilitiesBox"),
  agProbList: document.getElementById("agProbList"),

  // Waterlogging Panel
  wlStatusBadge: document.getElementById("wlStatusBadge"),
  wlErrorBox: document.getElementById("wlErrorBox"),
  wlContent: document.getElementById("wlContent"),
  wlWarningBanner: document.getElementById("wlWarningBanner"),
  wlRiskLevel: document.getElementById("wlRiskLevel"),
  wlProb: document.getElementById("wlProb"),
  wlTimelinePast24h: document.getElementById("wlTimelinePast24h"),
  wlTimelinePast7d: document.getElementById("wlTimelinePast7d"),
  wlTimelineFuture24h: document.getElementById("wlTimelineFuture24h"),
  wlTimelineFuture72h: document.getElementById("wlTimelineFuture72h"),
  wlRain1h: document.getElementById("wlRain1h"),
  wlRain3h: document.getElementById("wlRain3h"),
  wlRain6h: document.getElementById("wlRain6h"),
  wlRain12h: document.getElementById("wlRain12h"),
  wlRain24h: document.getElementById("wlRain24h"),
  wlRain3d: document.getElementById("wlRain3d"),
  wlRain7d: document.getElementById("wlRain7d"),
  wlFcNext1h: document.getElementById("wlFcNext1h"),
  wlFcNext3h: document.getElementById("wlFcNext3h"),
  wlFcNext6h: document.getElementById("wlFcNext6h"),
  wlFcNext12h: document.getElementById("wlFcNext12h"),
  wlFcNext24h: document.getElementById("wlFcNext24h"),
  wlFcNext48h: document.getElementById("wlFcNext48h"),
  wlFcNext72h: document.getElementById("wlFcNext72h"),
  wlElevation: document.getElementById("wlElevation"),
  wlSlope: document.getElementById("wlSlope"),
  wlTerrainSection: document.getElementById("wlTerrainSection"),
  wlTerrainStatus: document.getElementById("wlTerrainStatus"),
  wlRelativeElevation: document.getElementById("wlRelativeElevation"),
  wlLocalRelief: document.getElementById("wlLocalRelief"),
  wlGradient: document.getElementById("wlGradient"),
  wlDownslopeDirection: document.getElementById("wlDownslopeDirection"),
  wlProfileNS: document.getElementById("wlProfileNS"),
  wlProfileEW: document.getElementById("wlProfileEW"),
  wlOutlookBox: document.getElementById("wlOutlookBox"),
  wlOutlookBadge: document.getElementById("wlOutlookBadge"),
  wlOutlookSummary: document.getElementById("wlOutlookSummary"),
  wlMaxHourlyRain: document.getElementById("wlMaxHourlyRain"),
  wlTotalForecast72h: document.getElementById("wlTotalForecast72h"),
  wlMostSignificantPeriod: document.getElementById("wlMostSignificantPeriod"),
  wlDrainageBox: document.getElementById("wlDrainageBox"),
  wlDrainProximityBadge: document.getElementById("wlDrainProximityBadge"),
  wlDrainStatus: document.getElementById("wlDrainStatus"),
  wlDrainDistance: document.getElementById("wlDrainDistance"),
  wlDrainName: document.getElementById("wlDrainName"),
  wlDrainType: document.getElementById("wlDrainType"),
  wlDrainCount: document.getElementById("wlDrainCount"),
  wlDrainLength: document.getElementById("wlDrainLength"),
  wlDrainDensity: document.getElementById("wlDrainDensity"),
  wlDrainSourceNote: document.getElementById("wlDrainSourceNote"),
  wlDrainModelRisk: document.getElementById("wlDrainModelRisk"),
  wlDrainModelProb: document.getElementById("wlDrainModelProb"),
  wlDrainDataQualityBadge: document.getElementById("wlDrainDataQualityBadge"),
  wlDrainFeatureSummary: document.getElementById("wlDrainFeatureSummary"),
  wlDrainNdviVal: document.getElementById("wlDrainNdviVal"),
  wlDrainNdbiVal: document.getElementById("wlDrainNdbiVal"),
  wlDrainOpticalBadge: document.getElementById("wlDrainOpticalBadge"),
  wlDrainOpticalDate: document.getElementById("wlDrainOpticalDate"),
  wlDrainSarVal: document.getElementById("wlDrainSarVal"),
  wlDrainSarBadge: document.getElementById("wlDrainSarBadge"),
  wlDrainSarDate: document.getElementById("wlDrainSarDate"),
  wlSarVvVal: document.getElementById("wlSarVvVal"),
  wlSarVvBadge: document.getElementById("wlSarVvBadge"),
  wlSarVvMode: document.getElementById("wlSarVvMode"),
  wlS2StatusPill: document.getElementById("wlS2StatusPill"),
  wlS2SceneInfo: document.getElementById("wlS2SceneInfo"),

  // Spatial Rainfall Panel (Phase E Replacement)
  wlSpatialSection: document.getElementById("wlSpatialSection"),
  wlSpatialStatus: document.getElementById("wlSpatialStatus"),
  wlSpatialTime: document.getElementById("wlSpatialTime"),
  wlSpatialGridInfo: document.getElementById("wlSpatialGridInfo"),
  wlSpatial1h: document.getElementById("wlSpatial1h"),
  wlSpatial3h: document.getElementById("wlSpatial3h"),
  wlSpatial6h: document.getElementById("wlSpatial6h"),
  wlSpatial12h: document.getElementById("wlSpatial12h"),
  wlSpatial24h: document.getElementById("wlSpatial24h"),
  wlSpatial3d: document.getElementById("wlSpatial3d"),
  wlSpatial7d: document.getElementById("wlSpatial7d"),
  wlSpatialSourceNote: document.getElementById("wlSpatialSourceNote"),

  // Map Layer Controls (Phase E Replacement)
  layerItemRainfall: document.getElementById("layerItemRainfall"),
  layerToggleRainfall: document.getElementById("layerToggleRainfall"),
  layerBadgeRainfall: document.getElementById("layerBadgeRainfall"),

  // Workflow Open-Meteo Spatial Rainfall Provider Card
  sourceCardSpatialRainfall: document.getElementById("sourceCardSpatialRainfall"),
  sourceBadgeSpatialRainfall: document.getElementById("sourceBadgeSpatialRainfall"),
  sourceDescSpatialRainfall: document.getElementById("sourceDescSpatialRainfall"),

  // Sentinel-1 SAR Panel (Phase F)
  wlSarSection: document.getElementById("wlSarSection"),
  wlSarStatus: document.getElementById("wlSarStatus"),
  wlSarAcquisitionTime: document.getElementById("wlSarAcquisitionTime"),
  wlSarSceneAge: document.getElementById("wlSarSceneAge"),
  wlSarOrbitInfo: document.getElementById("wlSarOrbitInfo"),
  wlSarBaselineInfo: document.getElementById("wlSarBaselineInfo"),
  wlSarCandidateArea: document.getElementById("wlSarCandidateArea"),
  wlSarPersistentArea: document.getElementById("wlSarPersistentArea"),
  wlSarCoverage: document.getElementById("wlSarCoverage"),
  wlSarFraction: document.getElementById("wlSarFraction"),
  wlSarSourceNote: document.getElementById("wlSarSourceNote"),

  // Map Layer Controls (Phase F & Sentinel-2)
  layerItemSatellite: document.getElementById("layerItemSatellite"),
  layerToggleSatellite: document.getElementById("layerToggleSatellite"),
  layerBadgeSatellite: document.getElementById("layerBadgeSatellite"),
  layerItemSentinel2: document.getElementById("layerItemSentinel2"),
  layerToggleSentinel2: document.getElementById("layerToggleSentinel2"),
  layerBadgeSentinel2: document.getElementById("layerBadgeSentinel2"),

  // Workflow Sentinel-1 SAR Provider Card
  sourceCardSentinel1Sar: document.getElementById("sourceCardSentinel1Sar"),
  sourceBadgeSentinel1Sar: document.getElementById("sourceBadgeSentinel1Sar"),
  sourceDescSentinel1Sar: document.getElementById("sourceDescSentinel1Sar"),

  // Telemetry
  summaryProviderTime: document.getElementById("summaryProviderTime"),
  summaryTotalExecutionTime: document.getElementById("summaryTotalExecutionTime"),
  providerTelemetryBody: document.getElementById("providerTelemetryBody"),
  modulePerformanceBody: document.getElementById("modulePerformanceBody"),

  // Pan-India Hierarchical Spatial Summary Card Elements
  spatialSummaryCard: document.getElementById("spatialSummaryCard"),
  spatialRegionTitle: document.getElementById("spatialRegionTitle"),
  activeViewBadge: document.getElementById("activeViewBadge"),
  spatialSubTitle: document.getElementById("spatialSubTitle"),
  spatialCellCountBadge: document.getElementById("spatialCellCountBadge"),
  spatialDataQualityBadge: document.getElementById("spatialDataQualityBadge"),
  wlDistLow: document.getElementById("wlDistLow"),
  wlDistMod: document.getElementById("wlDistMod"),
  wlDistHigh: document.getElementById("wlDistHigh"),
  wlDistCrit: document.getElementById("wlDistCrit"),
  wlBarLow: document.getElementById("wlBarLow"),
  wlBarMod: document.getElementById("wlBarMod"),
  wlBarHigh: document.getElementById("wlBarHigh"),
  wlBarCrit: document.getElementById("wlBarCrit"),
  drDistLow: document.getElementById("drDistLow"),
  drDistMod: document.getElementById("drDistMod"),
  drDistHigh: document.getElementById("drDistHigh"),
  drDistCrit: document.getElementById("drDistCrit"),
  drBarLow: document.getElementById("drBarLow"),
  drBarMod: document.getElementById("drBarMod"),
  drBarHigh: document.getElementById("drBarHigh"),
  drBarCrit: document.getElementById("drBarCrit"),
  hotspotCountVal: document.getElementById("hotspotCountVal"),
  hotspotSubNote: document.getElementById("hotspotSubNote"),
  potentialCountVal: document.getElementById("potentialCountVal"),
  metaResolution: document.getElementById("metaResolution"),
  metaInferenceTime: document.getElementById("metaInferenceTime"),
  metaTotalLatency: document.getElementById("metaTotalLatency"),
  hotspotsSection: document.getElementById("hotspotsSection"),
  hotspotsTableBody: document.getElementById("hotspotsTableBody"),
};


// =============================================================================
// 2. HELPER UTILITIES
// =============================================================================

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatVal(val, unit = "", decimals = 2, fallback = "--") {
  if (val === undefined || val === null || val === "" || Number.isNaN(val)) {
    return fallback;
  }
  if (typeof val === "object") {
    return fallback;
  }
  if (typeof val === "number") {
    if (val === 0) {
      return unit ? `0 ${unit}` : "0";
    }
    const absVal = Math.abs(val);
    if (absVal < 0.01) {
      const p = absVal < 0.0001 ? 6 : 4;
      return unit ? `${val.toFixed(p)} ${unit}` : val.toFixed(p);
    }
    const formatted = Number.isInteger(val) ? val.toString() : val.toFixed(decimals);
    return unit ? `${formatted} ${unit}` : formatted;
  }
  return unit ? `${val} ${unit}` : String(val);
}

function getRiskBadgeClass(rawRisk) {
  if (!rawRisk || typeof rawRisk !== "string") return "badge-neutral";
  const r = rawRisk.trim().toUpperCase();
  if (r === "LOW" || r === "SAFE" || r === "NO STRESS" || r === "NO IRRIGATION" || r === "SUITABLE") return "badge-low";
  if (r === "MEDIUM" || r === "MODERATE") return "badge-moderate";
  if (r === "HIGH" || r === "HEAT STRESS" || r === "IRRIGATE" || r === "NOT SUITABLE") return "badge-high";
  if (r === "CRITICAL" || r === "SEVERE") return "badge-critical";
  return "badge-neutral";
}

function getRiskHexColor(rawRisk) {
  if (!rawRisk || typeof rawRisk !== "string") return "#64748b";
  const r = rawRisk.trim().toUpperCase();
  if (r === "LOW" || r === "SAFE") return "#10b981";
  if (r === "MEDIUM" || r === "MODERATE") return "#f59e0b";
  if (r === "HIGH") return "#f97316";
  if (r === "CRITICAL" || r === "SEVERE") return "#ef4444";
  return "#64748b";
}

// =============================================================================
// 3. LEAFLET MAP INITIALIZATION & LAYER CONTROLLER
// =============================================================================

function initMap() {
  if (mapState.map) return; // Initialize only once

  // Default view: India-wide [20.5937, 78.9629], zoom 5
  mapState.map = L.map("mapContainer", {
    center: [20.5937, 78.9629],
    zoom: 5,
    minZoom: 3,
    maxZoom: 18,
    zoomControl: true,
    attributionControl: false
  });

  // Base tile layer with fallback
  const baseTileLayer = L.tileLayer(BASEMAP_CONFIG.url, {
    attribution: BASEMAP_CONFIG.attribution,
    maxZoom: BASEMAP_CONFIG.maxZoom,
    subdomains: BASEMAP_CONFIG.subdomains
  });

  baseTileLayer.on("tileerror", () => {
    // If dark tiles fail, fallback gracefully to standard OSM
    if (!mapState.hasFallenBack) {
      mapState.hasFallenBack = true;
      L.tileLayer(BASEMAP_CONFIG.fallbackUrl, {
        attribution: BASEMAP_CONFIG.fallbackAttribution,
        maxZoom: 18
      }).addTo(mapState.map);
    }
  });

  baseTileLayer.addTo(mapState.map);

  // Legal Attribution in bottom-right
  L.control.attribution({ position: "bottomright", prefix: false })
    .addAttribution(BASEMAP_CONFIG.attribution)
    .addTo(mapState.map);

  // Initialize Layer Groups
  mapState.wlGridLayer = L.layerGroup().addTo(mapState.map);
  mapState.drainGridLayer = L.layerGroup().addTo(mapState.map);
  mapState.hotspotGridLayer = L.layerGroup().addTo(mapState.map);
  mapState.stateLayer = L.layerGroup().addTo(mapState.map);
  mapState.districtLayer = L.layerGroup().addTo(mapState.map);
  mapState.drainageLayer = L.layerGroup().addTo(mapState.map);
  mapState.drainageBufferCircle = L.layerGroup().addTo(mapState.map);
  mapState.terrainLayer = L.layerGroup().addTo(mapState.map);
  mapState.disasterLayer = L.layerGroup().addTo(mapState.map);
  mapState.fireLayer = L.layerGroup().addTo(mapState.map);
  mapState.alertsLayer = L.layerGroup().addTo(mapState.map);
  mapState.rainfallLayer = L.layerGroup().addTo(mapState.map);
  mapState.satelliteLayer = L.layerGroup().addTo(mapState.map);
  mapState.sentinel2Layer = L.layerGroup().addTo(mapState.map);

  // Bind Grid Layer Checkboxes
  if (dom.layerToggleWlGrid) {
    dom.layerToggleWlGrid.addEventListener("change", (e) => {
      mapState.activeLayers.wlGrid = e.target.checked;
      if (e.target.checked) {
        mapState.wlGridLayer.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.wlGridLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleDrainGrid) {
    dom.layerToggleDrainGrid.addEventListener("change", (e) => {
      mapState.activeLayers.drainGrid = e.target.checked;
      if (e.target.checked) {
        mapState.drainGridLayer.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.drainGridLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleHotspots) {
    dom.layerToggleHotspots.addEventListener("change", (e) => {
      mapState.activeLayers.hotspots = e.target.checked;
      if (e.target.checked) {
        mapState.hotspotGridLayer.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.hotspotGridLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleDrainage) {
    dom.layerToggleDrainage.addEventListener("change", (e) => {
      mapState.activeLayers.drainage = e.target.checked;
      if (e.target.checked) {
        mapState.drainageLayer.addTo(mapState.map);
        mapState.drainageBufferCircle.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.drainageLayer);
        mapState.map.removeLayer(mapState.drainageBufferCircle);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleSatellite) {
    dom.layerToggleSatellite.addEventListener("change", (e) => {
      mapState.activeLayers.satellite = e.target.checked;
      if (e.target.checked) {
        mapState.satelliteLayer.addTo(mapState.map);
        if (mapState.latestData.satellite && mapState.latestData.satellite.geojson) {
          renderSarWaterGeoJSON(mapState.latestData.satellite.geojson, selectedLocation.lat, selectedLocation.lon);
        }
      } else {
        mapState.map.removeLayer(mapState.satelliteLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleSentinel2) {
    dom.layerToggleSentinel2.addEventListener("change", (e) => {
      mapState.activeLayers.sentinel2 = e.target.checked;
      if (e.target.checked) {
        mapState.sentinel2Layer.addTo(mapState.map);
        if (mapState.latestData.sentinel2 && selectedLocation) {
          renderSentinel2MapMarker(mapState.latestData.sentinel2, selectedLocation.lat, selectedLocation.lon);
        }
      } else {
        mapState.map.removeLayer(mapState.sentinel2Layer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleRainfall) {
    dom.layerToggleRainfall.addEventListener("change", (e) => {
      mapState.activeLayers.rainfall = e.target.checked;
      if (e.target.checked) {
        mapState.rainfallLayer.addTo(mapState.map);
        if (mapState.latestData.rainfall && mapState.latestData.rainfall.grid && mapState.latestData.rainfall.grid.geojson) {
          renderRainfallGeoJSON(mapState.latestData.rainfall.grid.geojson, selectedLocation.lat, selectedLocation.lon);
        }
      } else {
        mapState.map.removeLayer(mapState.rainfallLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleDisaster) {
    dom.layerToggleDisaster.addEventListener("change", (e) => {
      mapState.activeLayers.disaster = e.target.checked;
      if (e.target.checked) {
        mapState.disasterLayer.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.disasterLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleFireHotspots) {
    dom.layerToggleFireHotspots.addEventListener("change", (e) => {
      mapState.activeLayers.fireHotspots = e.target.checked;
      if (e.target.checked) {
        mapState.fireLayer.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.fireLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleActiveAlerts) {
    dom.layerToggleActiveAlerts.addEventListener("change", (e) => {
      mapState.activeLayers.activeAlerts = e.target.checked;
      if (e.target.checked) {
        mapState.alertsLayer.addTo(mapState.map);
      } else {
        mapState.map.removeLayer(mapState.alertsLayer);
      }
      updateLegend();
    });
  }

  if (dom.layerToggleWaterlogging) {
    dom.layerToggleWaterlogging.addEventListener("change", (e) => {
      mapState.activeLayers.waterlogging = e.target.checked;
      if (mapState.selectedMarker) {
        if (e.target.checked) {
          mapState.selectedMarker.addTo(mapState.map);
        } else {
          mapState.map.removeLayer(mapState.selectedMarker);
        }
      }
      updateLegend();
    });
  }

  // Hierarchical View Buttons
  if (dom.viewBtnIndia) {
    dom.viewBtnIndia.addEventListener("click", () => loadIndiaView());
  }
  if (dom.viewBtnState) {
    dom.viewBtnState.addEventListener("click", () => {
      const st = (dom.navStateSelect && dom.navStateSelect.value) || "Bihar";
      loadStateView(st);
    });
  }
  if (dom.viewBtnDistrict) {
    dom.viewBtnDistrict.addEventListener("click", () => {
      const st = (dom.navStateSelect && dom.navStateSelect.value) || "Bihar";
      const dst = (dom.navDistrictSelect && dom.navDistrictSelect.value) || "Patna";
      loadDistrictView(st, dst);
    });
  }
  if (dom.viewBtnGrid) {
    dom.viewBtnGrid.addEventListener("click", () => {
      const res = Number((dom.navGridResolution && dom.navGridResolution.value) || 500);
      generateLocalRiskGrid(selectedLocation.lat, selectedLocation.lon, 5000, res);
    });
  }

  if (dom.navStateSelect) {
    dom.navStateSelect.addEventListener("change", (e) => {
      if (e.target.value) loadStateView(e.target.value);
    });
  }

  if (dom.navDistrictSelect) {
    dom.navDistrictSelect.addEventListener("change", (e) => {
      if (e.target.value && dom.navStateSelect && dom.navStateSelect.value) {
        loadDistrictView(dom.navStateSelect.value, e.target.value);
      }
    });
  }

  if (dom.btnGenerateGrid) {
    dom.btnGenerateGrid.addEventListener("click", () => {
      const res = Number((dom.navGridResolution && dom.navGridResolution.value) || 500);
      generateLocalRiskGrid(selectedLocation.lat, selectedLocation.lon, 5000, res);
    });
  }

  if (dom.mapResetBtn) {
    dom.mapResetBtn.addEventListener("click", () => loadIndiaView());
  }

  // Resize handler to prevent blank/gray tiles
  window.addEventListener("resize", () => {
    if (mapState.map) {
      mapState.map.invalidateSize();
    }
  });

  updateLegend();
}

// Update Dynamic Map Legend based only on active layers
function updateLegend() {
  if (!dom.legendContent) return;
  dom.legendContent.innerHTML = "";

  // 1. Waterlogging Risk Grid Section
  if (mapState.activeLayers.wlGrid) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Waterlogging Risk Grid</div>
      <div class="legend-items-row">
        <span class="legend-item"><span class="legend-color-dot" style="background:#10b981;"></span> Low</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f59e0b;"></span> Moderate</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f97316;"></span> High</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#ef4444;"></span> Critical</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#64748b;"></span> No Data</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 2. Drainage Risk Grid Section
  if (mapState.activeLayers.drainGrid) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Drainage Risk Grid</div>
      <div class="legend-items-row">
        <span class="legend-item"><span class="legend-color-dot" style="background:#10b981;"></span> Adequate</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f59e0b;"></span> Moderate</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f97316;"></span> High Congestion</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#ef4444;"></span> Critical</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#64748b;"></span> No Data</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 3. Combined Hotspots Section
  if (mapState.activeLayers.hotspots) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Combined Risk Hotspots</div>
      <div class="legend-items-row">
        <span class="legend-item"><span class="legend-color-dot" style="background:#ef4444; border: 1.5px dashed #991b1b;"></span> Critical Hotspot</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f97316; border: 1px dashed #c2410c;"></span> High-Risk Hotspot</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f59e0b; border: 1px dashed #d97706;"></span> Potential Hotspot</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 4. OSM Drainage Network Section
  if (mapState.activeLayers.drainage) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">OSM Drainage Network</div>
      <div class="legend-items-row">
        <span class="legend-item"><span class="legend-line-sample" style="background:#38bdf8;"></span> River</span>
        <span class="legend-item"><span class="legend-line-sample" style="background:#06b6d4;"></span> Stream</span>
        <span class="legend-item"><span class="legend-line-sample" style="background:#818cf8; border-top: 1px dashed #818cf8;"></span> Canal</span>
        <span class="legend-item"><span class="legend-line-sample" style="background:#a78bfa;"></span> Drain</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 5. Sentinel-1 SAR Surface Water
  if (mapState.activeLayers.satellite) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Sentinel-1 SAR Surface Water</div>
      <div class="legend-items-row" style="flex-wrap: wrap; gap: 6px;">
        <span class="legend-item"><span class="legend-color-dot" style="background:#ef4444; border:1px solid #b91c1c;"></span> Possible New Water</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#0284c7; border:1px solid #0369a1;"></span> Persistent Low Backscatter</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 6. Sentinel-2 Optical
  if (mapState.activeLayers.sentinel2) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Sentinel-2 Optical (NDVI / NDBI)</div>
      <div class="legend-items-row" style="flex-wrap: wrap; gap: 6px;">
        <span class="legend-item"><span class="legend-color-dot" style="background:#22c55e;"></span> High Vegetation (NDVI &gt; 0.4)</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#cbd5e1;"></span> Built-up Surface (NDBI &gt; 0.0)</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 7. Spatial Rainfall Grid Section
  if (mapState.activeLayers.rainfall) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Spatial Rainfall (24h mm)</div>
      <div class="legend-items-row" style="flex-wrap: wrap; gap: 5px;">
        <span class="legend-item"><span class="legend-color-dot" style="background:rgba(56,189,248,0.3); border:1px solid #38bdf8;"></span> &lt;5mm</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:rgba(14,165,233,0.5); border:1px solid #0ea5e9;"></span> 5–15mm</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:rgba(37,99,235,0.7); border:1px solid #2563eb;"></span> 15–30mm</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:rgba(124,58,237,0.8); border:1px solid #7c3aed;"></span> 30–60mm</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:rgba(225,29,72,0.9); border:1px solid #e11d48;"></span> 60mm+</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 8. Disaster Events Section
  if (mapState.activeLayers.disaster) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Live Disaster Hazards</div>
      <div class="legend-items-row">
        <span class="legend-item"><span class="legend-color-dot" style="background:#f43f5e;"></span> Quake Epicenter</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#fbbf24;"></span> GDACS Alert</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 9. Active Wildfire Hotspots Section
  if (mapState.activeLayers.fireHotspots) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Active Wildfires (NASA FIRMS)</div>
      <div class="legend-items-row" style="flex-wrap: wrap; gap: 5px;">
        <span class="legend-item"><span class="legend-color-dot" style="background:#ef4444;"></span> High FRP (&ge;50 MW)</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f97316;"></span> Moderate FRP (25–50 MW)</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#fbbf24;"></span> Thermal Hotspot (&lt;25 MW)</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }

  // 10. Active Operational Alerts Section
  if (mapState.activeLayers.activeAlerts) {
    const sec = document.createElement("div");
    sec.className = "legend-section";
    sec.innerHTML = `
      <div class="legend-sec-title">Active Operational Alerts</div>
      <div class="legend-items-row" style="flex-wrap: wrap; gap: 5px;">
        <span class="legend-item"><span class="legend-color-dot" style="background:#ef4444;"></span> Critical / Severe</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#f97316;"></span> High</span>
        <span class="legend-item"><span class="legend-color-dot" style="background:#38bdf8;"></span> Moderate / Low</span>
      </div>
    `;
    dom.legendContent.appendChild(sec);
  }
}



// Update or create the single authoritative Selected Location Marker
function updateSelectedLocationMarker(data) {
  if (!mapState.map) return;

  const lat = selectedLocation.lat;
  const lon = selectedLocation.lon;
  const placeName = selectedLocation.name || `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;

  const wl = (data && data.waterlogging && data.waterlogging.data) || {};
  const wlRes = wl.result || {};
  const riskLevel = wlRes.risk_level || "LOW";
  const riskColor = getRiskHexColor(riskLevel);
  const posProbStr = wlRes.positive_probability !== undefined ? `${(wlRes.positive_probability * 100).toFixed(1)}%` : "--";
  const warningActive = wlRes.warning === true;

  // Weather summary
  const wPred = (data && data.weather && data.weather.data && data.weather.data.result && data.weather.data.result.prediction) || {};
  const tempStr = formatVal(wPred.temperature_2m, "°C");
  const rain24hStr = formatVal(wPred.rain, "mm");

  // Forecast summary
  const fc = wlRes.forecast_rainfall || (wl.input && wl.input.forecast_rainfall) || {};
  const fc24hStr = formatVal(fc.next_24h && fc.next_24h.value, "mm");

  // Disaster summary (Verified hazard prioritized)
  const dData = (data && data.disaster && data.disaster.data) || {};
  const vHaz = dData.verified_hazard || (dData.result && dData.result.verified_hazard) || null;
  let dSummary = "No active alert";
  if (vHaz && vHaz.active && vHaz.hazard) {
    dSummary = `Verified ${vHaz.hazard.toUpperCase().replace(/_/g, " ")}`;
  } else if (vHaz && vHaz.status === "UNCORROBORATED_MODEL_SIGNAL") {
    dSummary = `Uncorroborated (${dData.raw_model?.prediction || dData.result?.prediction || "ML"})`;
  } else {
    dSummary = "No active alert";
  }

  // Agriculture summary
  const agPred = (data && data.agriculture && data.agriculture.data && data.agriculture.data.result && data.agriculture.data.result.prediction) || {};
  const agRisk = agPred.crop_weather_risk || "Low";

  // Custom DivIcon with pulsing radar beacon
  const iconHtml = `
    <div class="pulse-beacon-marker" style="color: ${riskColor};">
      <div class="beacon-ripple" style="background: ${riskColor};"></div>
      <div class="beacon-core" style="background: ${riskColor};"></div>
    </div>
  `;

  const customIcon = L.divIcon({
    html: iconHtml,
    className: "custom-beacon-div-icon",
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -14]
  });

  // Build safe popup content
  const popupHtml = `
    <div class="popup-title">${escapeHtml(placeName)}</div>
    <div class="popup-coords">${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</div>
    <div class="disclaimer-tag" style="margin-bottom: 6px;">Point analysis at selected/resolved coordinate</div>
    <div class="popup-grid">
      <div class="popup-cell"><strong>Waterlogging:</strong> ${escapeHtml(riskLevel)} (${posProbStr})</div>
      <div class="popup-cell"><strong>Warning (0.36):</strong> ${warningActive ? '<span style="color:#ef4444;">ACTIVE</span>' : '<span style="color:#10b981;">NO</span>'}</div>
      <div class="popup-cell"><strong>Temp:</strong> ${tempStr}</div>
      <div class="popup-cell"><strong>Rain (24h/Next 24h):</strong> ${rain24hStr} / ${fc24hStr}</div>
      <div class="popup-cell"><strong>Disaster:</strong> ${escapeHtml(dSummary)}</div>
      <div class="popup-cell"><strong>Agri Risk:</strong> ${escapeHtml(agRisk)}</div>
    </div>
    <div class="popup-footer-note">Smart Multi-Risk Platform • Pan-India Local Intelligence</div>
  `;

  if (mapState.selectedMarker) {
    mapState.selectedMarker.setLatLng([lat, lon]);
    mapState.selectedMarker.setIcon(customIcon);
    mapState.selectedMarker.getPopup().setContent(popupHtml);
  } else {
    mapState.selectedMarker = L.marker([lat, lon], { icon: customIcon })
      .bindPopup(popupHtml, { maxWidth: 320 });
    if (mapState.activeLayers.waterlogging) {
      mapState.selectedMarker.addTo(mapState.map);
    }
  }

  // Smooth pan/zoom to location
  mapState.map.flyTo([lat, lon], 12, { duration: 1.0 });
}

// Render Drainage GeoJSON Layer from Phase B
function renderDrainageGeoJSON(d, reqLat, reqLon) {
  if (!mapState.map || !mapState.drainageLayer) return;

  // Clear existing drainage layers
  mapState.drainageLayer.clearLayers();
  mapState.drainageBufferCircle.clearLayers();

  if (!d) return;

  // Render 2000m Drainage Analysis Radius Circle
  const bufferCircle = L.circle([reqLat, reqLon], {
    radius: d.analysis_radius_m || 2000,
    color: "#38bdf8",
    weight: 1.5,
    dashArray: "4, 6",
    fillColor: "#38bdf8",
    fillOpacity: 0.03
  }).bindTooltip("Drainage analysis radius (2km buffer)", { permanent: false, direction: "top" });

  mapState.drainageBufferCircle.addLayer(bufferCircle);

  // If no mapped data or unavailable, return without lines
  if (d.status !== "ok" || !d.geojson || !Array.isArray(d.geojson.features)) {
    return;
  }

  // Color mapping by waterway type
  const colorMap = {
    river: "#38bdf8",
    stream: "#06b6d4",
    canal: "#818cf8",
    drain: "#a78bfa",
    ditch: "#c084fc"
  };

  L.geoJSON(d.geojson, {
    style: function (feature) {
      const kind = (feature.properties && feature.properties.waterway) || "unknown";
      const isCanal = kind === "canal";
      return {
        color: colorMap[kind] || "#94a3b8",
        weight: kind === "river" ? 3.5 : (kind === "stream" ? 2.5 : 2.0),
        opacity: 0.85,
        dashArray: isCanal ? "4, 4" : null
      };
    },
    onEachFeature: function (feature, layer) {
      const p = feature.properties || {};
      const name = p.name ? escapeHtml(p.name) : "Unnamed Waterway";
      const kind = p.waterway ? escapeHtml(p.waterway) : "unknown";
      const inBufLen = p.in_buffer_length_m !== undefined ? `${p.in_buffer_length_m.toFixed(1)} m` : "--";
      const osmId = p.osm_id ? escapeHtml(p.osm_id) : "--";

      layer.bindPopup(`
        <div class="popup-title">${name}</div>
        <div class="popup-coords">Type: ${kind.toUpperCase()}</div>
        <div class="popup-cell"><strong>In-Buffer Length:</strong> ${inBufLen}</div>
        <div class="popup-cell"><strong>OSM ID:</strong> ${osmId}</div>
        <div class="popup-footer-note">Source: OpenStreetMap via Overpass API</div>
      `, { maxWidth: 260 });
    }
  }).addTo(mapState.drainageLayer);
}

// Render Spatial Rainfall 0.1° Grid GeoJSON (Phase E Replacement)
function renderRainfallGeoJSON(geojson, reqLat, reqLon) {
  if (!mapState.map || !mapState.rainfallLayer) return;

  // Clear existing rainfall layer
  mapState.rainfallLayer.clearLayers();

  if (!geojson || !Array.isArray(geojson.features) || !mapState.activeLayers.rainfall) {
    return;
  }

  // DISPLAY COLOR BREAKS (Continuous numeric precipitation scale for 24h accumulation mm; NOT risk/warning thresholds)
  function getRainfallColor(mm) {
    if (mm === null || mm === undefined) return "rgba(148, 163, 184, 0.15)";
    if (mm <= 0.0) return "rgba(56, 189, 248, 0.08)";
    if (mm < 5.0) return "rgba(56, 189, 248, 0.35)";
    if (mm < 15.0) return "rgba(14, 165, 233, 0.55)";
    if (mm < 30.0) return "rgba(37, 99, 235, 0.70)";
    if (mm < 60.0) return "rgba(124, 58, 237, 0.80)";
    return "rgba(225, 29, 72, 0.90)";
  }

  function getRainfallBorderColor(mm) {
    if (mm === null || mm === undefined) return "#64748b";
    if (mm <= 0.0) return "#38bdf8";
    if (mm < 5.0) return "#38bdf8";
    if (mm < 15.0) return "#0ea5e9";
    if (mm < 30.0) return "#2563eb";
    if (mm < 60.0) return "#7c3aed";
    return "#e11d48";
  }

  L.geoJSON(geojson, {
    style: function (feature) {
      const mm = feature.properties ? feature.properties.precipitation_mm : null;
      return {
        fillColor: getRainfallColor(mm),
        fillOpacity: mm === null ? 0.1 : 0.65,
        color: getRainfallBorderColor(mm),
        weight: 1.0,
        opacity: 0.8,
      };
    },
    onEachFeature: function (feature, layer) {
      const p = feature.properties || {};
      const mmStr = (p.precipitation_mm !== null && p.precipitation_mm !== undefined)
        ? `${p.precipitation_mm.toFixed(2)} mm`
        : "No Data / Missing";
      const sLat = p.sampling_center_lat !== undefined ? p.sampling_center_lat : (p.cell_center_lat || "--");
      const sLon = p.sampling_center_lon !== undefined ? p.sampling_center_lon : (p.cell_center_lon || "--");
      const retLat = p.provider_returned_lat !== undefined ? `${p.provider_returned_lat}°N` : "--";
      const retLon = p.provider_returned_lon !== undefined ? `${p.provider_returned_lon}°E` : "--";
      const win = p.time_window || "rolling_24h";

      layer.bindPopup(`
        <div class="popup-title">Spatial Rainfall Visualization Cell</div>
        <div class="popup-coords">Sampling Point: ${sLat}°N, ${sLon}°E (0.1° request spacing)</div>
        <div class="popup-grid">
          <div class="popup-cell"><strong>Accumulation (24h):</strong> ${mmStr}</div>
          <div class="popup-cell"><strong>Window:</strong> ${win}</div>
          <div class="popup-cell"><strong>Provider Return:</strong> ${retLat}, ${retLon}</div>
          <div class="popup-cell"><strong>Status:</strong> ${(p.status || "valid").toUpperCase()}</div>
        </div>
        <div class="popup-footer-note">Source: Open-Meteo Recent Hourly Model Data (Best Match NWP)</div>
      `, { maxWidth: 280 });
    }
  }).addTo(mapState.rainfallLayer);
}

// Render Sentinel-1 SAR GeoJSON Layer (Phase F)
function renderSarWaterGeoJSON(geojson, reqLat, reqLon) {
  if (!mapState.map || !mapState.satelliteLayer) return;

  // Clear existing satellite layers
  mapState.satelliteLayer.clearLayers();

  if (!geojson || !Array.isArray(geojson.features) || !mapState.activeLayers.satellite) {
    return;
  }

  L.geoJSON(geojson, {
    style: function (feature) {
      const p = feature.properties || {};
      const cls = p.class || "possible_new_surface_water";
      if (cls === "possible_new_surface_water") {
        return {
          fillColor: "#ef4444",
          fillOpacity: 0.65,
          color: "#b91c1c",
          weight: 1.5,
          opacity: 0.9
        };
      } else {
        return {
          fillColor: "#0284c7",
          fillOpacity: 0.45,
          color: "#0369a1",
          weight: 1.0,
          opacity: 0.8
        };
      }
    },
    onEachFeature: function (feature, layer) {
      const p = feature.properties || {};
      const isNew = p.class === "possible_new_surface_water";
      const title = isNew ? "Possible New Surface Water" : "Persistent Low Backscatter";
      const areaKm2 = p.area_km2 !== undefined ? `${p.area_km2.toFixed(4)} km²` : "--";
      const areaM2 = p.area_m2 !== undefined ? `${Math.round(p.area_m2).toLocaleString()} m²` : "--";
      const acqTime = escapeHtml(p.recent_acquisition_time || "--");
      const ageHours = p.scene_age_hours !== undefined ? `${p.scene_age_hours.toFixed(1)} h` : "--";
      const sceneId = escapeHtml(p.recent_scene_id || "--");
      const orbitDir = escapeHtml(p.orbit_direction || "--");
      const relOrbit = p.relative_orbit !== undefined ? p.relative_orbit : "--";
      const baseCount = p.baseline_scene_count !== undefined ? p.baseline_scene_count : "--";
      const method = escapeHtml(p.method || "SAR temporal backscatter change");

      layer.bindPopup(`
        <div class="popup-title" style="color: ${isNew ? '#ef4444' : '#38bdf8'};">${title}</div>
        <div class="popup-coords">Area: ${areaKm2} (${areaM2})</div>
        <div class="popup-grid">
          <div class="popup-cell"><strong>Acquisition:</strong> ${acqTime}</div>
          <div class="popup-cell"><strong>Scene Age:</strong> ${ageHours}</div>
          <div class="popup-cell"><strong>Orbit:</strong> ${orbitDir} (Rel: ${relOrbit})</div>
          <div class="popup-cell"><strong>Baseline Scenes:</strong> ${baseCount}</div>
          <div class="popup-cell" style="grid-column: span 2;"><strong>Scene ID:</strong> <span style="font-family: monospace; font-size: 11px;">${sceneId}</span></div>
          <div class="popup-cell" style="grid-column: span 2;"><strong>Method:</strong> ${method}</div>
        </div>
        <div class="popup-footer-note">
          Sentinel-1 SAR change analysis identifies radar signatures consistent with possible new surface water. Results are contextual and may include false positives from smooth surfaces, agriculture, radar shadow, or other backscatter changes.
        </div>
      `, { maxWidth: 320 });
    }
  }).addTo(mapState.satelliteLayer);
}


// Render Terrain Sample Points GeoJSON from Phase C
function renderTerrainSamplePoints(geojson, downslopeDir, reqLat, reqLon) {
  if (!mapState.map || !mapState.terrainLayer) return;

  mapState.terrainLayer.clearLayers();
  if (!geojson || !Array.isArray(geojson.features)) return;

  geojson.features.forEach((feat) => {
    const coords = feat.geometry && feat.geometry.coordinates;
    if (!coords || coords.length < 2) return;
    const lon = coords[0];
    const lat = coords[1];
    const p = feat.properties || {};

    const isModel = p.sample_role === "model" || p.sample_id === "model_center";
    const sampleId = p.sample_id ? escapeHtml(p.sample_id) : "--";
    const role = isModel ? "Model Stencil Point" : "1km Contextual Ring";
    const dir = p.direction ? escapeHtml(p.direction) : "--";
    const dist = p.distance_m !== undefined ? `${p.distance_m.toFixed(0)} m` : "--";
    const elev = p.elevation_m !== null && p.elevation_m !== undefined ? `${p.elevation_m.toFixed(1)} m` : "--";

    const popupHtml = `
      <div class="popup-title">${sampleId} (${dir})</div>
      <div class="popup-coords">${role}</div>
      <div class="popup-cell"><strong>Elevation:</strong> ${elev}</div>
      <div class="popup-cell"><strong>Distance:</strong> ${dist}</div>
      <div class="popup-footer-note">Source: Open-Meteo Elevation API (No fake heatmap)</div>
    `;

    if (isModel) {
      // Model points rendered as cyan diamonds
      const diamondIcon = L.divIcon({
        html: `<div class="terrain-model-glyph"></div>`,
        className: "custom-terrain-glyph",
        iconSize: [12, 12],
        iconAnchor: [6, 6]
      });
      L.marker([lat, lon], { icon: diamondIcon })
        .bindPopup(popupHtml, { maxWidth: 240 })
        .addTo(mapState.terrainLayer);
    } else {
      // Context ring points rendered as purple circles
      L.circleMarker([lat, lon], {
        radius: 5,
        color: "#a855f7",
        weight: 1.5,
        fillColor: "#a855f7",
        fillOpacity: 0.85
      })
        .bindPopup(popupHtml, { maxWidth: 240 })
        .addTo(mapState.terrainLayer);
    }
  });

  // Downslope Direction Indicator at Center
  if (downslopeDir && downslopeDir.cardinal !== "FLAT" && downslopeDir.azimuth_deg !== undefined) {
    const arrowIcon = L.divIcon({
      html: `<div class="downslope-arrow-indicator" title="Approximate Local Terrain Downslope Direction: ${downslopeDir.cardinal} (${downslopeDir.azimuth_deg}°)">${escapeHtml(downslopeDir.arrow || '↓')}</div>`,
      className: "custom-downslope-glyph",
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });
    L.marker([reqLat, reqLon], { icon: arrowIcon })
      .bindTooltip(`Approximate Local Terrain Downslope Direction: ${downslopeDir.cardinal} (${downslopeDir.azimuth_deg}°)`, { permanent: false })
      .addTo(mapState.terrainLayer);
  }
}

// Render Real Disaster Event Markers (USGS & GDACS coordinates only)
function renderDisasterEventMarkers(disasterData) {
  if (!mapState.map || !mapState.disasterLayer) return;
  mapState.disasterLayer.clearLayers();

  if (!disasterData || !disasterData.data) return;

  const input = disasterData.data.input || {};
  const seismic = input.seismic_metadata || {};
  const gdacs = input.gdacs_metadata || {};

  // 1. USGS Earthquake Marker (Real coordinates only)
  if (seismic.event_lat !== null && seismic.event_lat !== undefined &&
    seismic.event_lon !== null && seismic.event_lon !== undefined &&
    seismic.max_quake_mag && seismic.max_quake_mag > 0) {

    const qLat = Number(seismic.event_lat);
    const qLon = Number(seismic.event_lon);
    const mag = Number(seismic.max_quake_mag);
    const depth = formatVal(seismic.quake_depth, "km");
    const dist = formatVal(seismic.event_distance_km, "km");
    const title = escapeHtml(seismic.event_title || "Seismic Event");
    const timeUtc = escapeHtml(seismic.event_time_utc || "--");

    const radius = Math.max(8, Math.min(24, mag * 3));

    L.circleMarker([qLat, qLon], {
      radius: radius,
      color: "#f43f5e",
      weight: 2,
      fillColor: "#f43f5e",
      fillOpacity: 0.45
    })
      .bindPopup(`
        <div class="popup-title">USGS Earthquake Epicenter</div>
        <div class="popup-coords">Magnitude: M${mag.toFixed(1)} • Depth: ${depth}</div>
        <div class="popup-cell"><strong>Event:</strong> ${title}</div>
        <div class="popup-cell"><strong>Distance:</strong> ${dist}</div>
        <div class="popup-cell"><strong>UTC Time:</strong> ${timeUtc}</div>
        <div class="popup-footer-note">Source: USGS Earthquake Hazards Program</div>
      `, { maxWidth: 280 })
      .addTo(mapState.disasterLayer);
  }

  // 2. GDACS Active Event Markers (Real coordinates only)
  const auditEvents = gdacs.audit_events || [];
  auditEvents.forEach((ev) => {
    if (ev.active && ev.geographically_relevant && ev.lat && ev.lon) {
      const eLat = Number(ev.lat);
      const eLon = Number(ev.lon);
      const name = escapeHtml(ev.event_name || "Hazard Event");
      const etype = escapeHtml(ev.event_type || "Hazard");
      const dist = formatVal(ev.distance_km, "km");

      L.circleMarker([eLat, eLon], {
        radius: 10,
        color: "#fbbf24",
        weight: 2,
        fillColor: "#fbbf24",
        fillOpacity: 0.6
      })
        .bindPopup(`
          <div class="popup-title">GDACS Active Alert</div>
          <div class="popup-coords">${etype}: ${name}</div>
          <div class="popup-cell"><strong>Distance:</strong> ${dist}</div>
          <div class="popup-cell"><strong>Status:</strong> ACTIVE / GEOGRAPHICALLY RELEVANT</div>
          <div class="popup-footer-note">Source: GDACS Active Hazard Feeds</div>
        `, { maxWidth: 260 })
        .addTo(mapState.disasterLayer);
    }
  });

  // 3. Render NASA FIRMS Active Fire Hotspots
  const dData = disasterData.data || {};
  const fireHotspots = dData.fire_hotspots || (dData.live_evidence && dData.live_evidence.fire_hotspots) || [];
  const reqLat = dData.location ? dData.location.lat : (selectedLocation ? selectedLocation.lat : null);
  const reqLon = dData.location ? dData.location.lon : (selectedLocation ? selectedLocation.lon : null);
  renderFireHotspotMarkers(fireHotspots, reqLat, reqLon);

  // 4. Render Operational Active Alerts
  const activeAlerts = dData.active_alerts || (dData.alerts && dData.alerts.active_alerts) || [];
  renderOperationalAlertMarkers(activeAlerts);
}

// Render NASA FIRMS Active Fire Hotspots Layer
function renderFireHotspotMarkers(hotspots, selLat, selLon) {
  if (!mapState.map || !mapState.fireLayer) return;
  mapState.fireLayer.clearLayers();

  if (!Array.isArray(hotspots) || hotspots.length === 0) return;

  hotspots.forEach((h) => {
    if (h.latitude == null || h.longitude == null) return;
    const hLat = Number(h.latitude);
    const hLon = Number(h.longitude);
    if (isNaN(hLat) || isNaN(hLon)) return;

    const frp = Number(h.frp || 0.0);
    const sat = escapeHtml(h.satellite || "VIIRS NOAA-21");
    const inst = escapeHtml(h.instrument || "VIIRS");
    const acqDate = escapeHtml(h.acq_date || "--");
    const acqTime = escapeHtml(h.acq_time || "--");
    const conf = escapeHtml(String(h.confidence || "nominal"));
    const daynight = h.daynight === "D" ? "Day" : (h.daynight === "N" ? "Night" : escapeHtml(String(h.daynight || "--")));

    let distStr = "--";
    if (h.distance_km != null) {
      distStr = `${Number(h.distance_km).toFixed(1)} km`;
    } else if (selLat != null && selLon != null) {
      const dLat = (hLat - selLat) * 111.0;
      const dLon = (hLon - selLon) * 111.0 * Math.cos(selLat * Math.PI / 180);
      distStr = `${Math.sqrt(dLat * dLat + dLon * dLon).toFixed(1)} km`;
    }

    // Severity-aware styling consistent with existing UI
    let markerColor = "#fbbf24"; // moderate / amber
    let radius = 7;
    if (frp >= 50.0) {
      markerColor = "#ef4444"; // critical / intense
      radius = 12;
    } else if (frp >= 25.0) {
      markerColor = "#f97316"; // high / orange
      radius = 9.5;
    }

    const popupHtml = `
      <div class="popup-title">🔥 Active Wildfire Hotspot</div>
      <div class="popup-coords">Satellite: ${sat} (${inst})</div>
      <div class="popup-cell"><strong>Detection Time:</strong> ${acqDate} ${acqTime} UTC (${daynight})</div>
      <div class="popup-cell"><strong>Confidence:</strong> ${conf}</div>
      <div class="popup-cell"><strong>Fire Radiative Power (FRP):</strong> ${frp.toFixed(1)} MW</div>
      <div class="popup-cell"><strong>Distance:</strong> ${distStr}</div>
      <div class="popup-footer-note">Source: NASA FIRMS (VIIRS NRT)</div>
    `;

    L.circleMarker([hLat, hLon], {
      radius: radius,
      color: markerColor,
      weight: 2,
      fillColor: markerColor,
      fillOpacity: 0.7
    })
      .bindPopup(popupHtml, { maxWidth: 280 })
      .addTo(mapState.fireLayer);
  });
}

// Render Operational Active Alerts on Map (Section 16)
function renderOperationalAlertMarkers(alerts) {
  if (!mapState.map || !mapState.alertsLayer) return;
  mapState.alertsLayer.clearLayers();
  if (!Array.isArray(alerts) || alerts.length === 0) return;

  alerts.forEach(a => {
    if (a.lat == null || a.lon == null) return;
    const aLat = Number(a.lat);
    const aLon = Number(a.lon);
    if (isNaN(aLat) || isNaN(aLon)) return;

    const sev = (a.severity || "LOW").toUpperCase();
    const stat = (a.status || "ACTIVE").toUpperCase();
    const hzName = (a.hazard || "Disaster").toUpperCase().replace(/_/g, ' ');
    const locStr = [a.district, a.state].filter(Boolean).join(", ") || "Regional";
    const sources = (a.sources || []).join(", ") || "Multi-sensor";
    const started = a.started_at ? new Date(a.started_at).toLocaleString() : "--";
    const verified = a.last_verified_at ? new Date(a.last_verified_at).toLocaleString() : "--";

    let color = "#38bdf8";
    if (sev === "CRITICAL" || sev === "SEVERE") color = "#ef4444";
    else if (sev === "HIGH") color = "#f97316";
    else if (sev === "MODERATE") color = "#f59e0b";

    const popupHtml = `
      <div class="popup-title">${getHazardIcon(a.hazard)} ${escapeHtml(hzName)} — ${escapeHtml(sev)}</div>
      <div class="popup-coords">${escapeHtml(locStr)}</div>
      <div class="popup-cell"><strong>Status:</strong> <span class="alert-lifecycle-badge alert-lifecycle-${stat}">${stat}</span></div>
      <div class="popup-cell"><strong>Started:</strong> ${started}</div>
      <div class="popup-cell"><strong>Last Verified:</strong> ${verified}</div>
      <div class="popup-cell"><strong>Sources:</strong> ${escapeHtml(sources)}</div>
      <div class="popup-cell" style="margin-top: 4px; font-size: 10px; color: #cbd5e1;">${escapeHtml(a.description || '')}</div>
    `;

    L.circleMarker([aLat, aLon], {
      radius: 12,
      color: color,
      weight: 3,
      fillColor: color,
      fillOpacity: 0.55
    })
      .bindPopup(popupHtml, { maxWidth: 280 })
      .addTo(mapState.alertsLayer);
  });
}

function getHazardIcon(h) {
  const hz = (h || "").toLowerCase();
  if (hz === "flood" || hz === "flash_flood") return "🌊";
  if (hz === "wildfire") return "🔥";
  if (hz === "avalanche") return "🏔️";
  if (hz === "earthquake") return "⚡";
  if (hz === "cyclone") return "🌀";
  if (hz === "thunderstorm") return "⛈️";
  if (hz === "high_wind") return "💨";
  if (hz === "heat_wave") return "🌡️";
  if (hz === "cold_wave") return "❄️";
  if (hz === "drought") return "🌾";
  if (hz === "air_pollution") return "🌫️";
  if (hz === "tsunami") return "🌊";
  if (hz === "volcano") return "🌋";
  if (hz === "landslide") return "⛰️";
  return "⚠️";
}

// Render Active Operational Alerts & Lifecycle History (Section 15)
function renderActiveAlertsAndHistory(alertsData, location) {
  const active = alertsData.active_alerts || [];

  if (dom.activeAlertsCountBadge) {
    if (active.length > 0) {
      dom.activeAlertsCountBadge.textContent = `${active.length} ACTIVE`;
      dom.activeAlertsCountBadge.className = "status-pill status-danger";
    } else {
      dom.activeAlertsCountBadge.textContent = "0 ACTIVE";
      dom.activeAlertsCountBadge.className = "status-pill status-neutral";
    }
  }

  if (dom.activeAlertsList) {
    dom.activeAlertsList.innerHTML = "";
    if (active.length === 0) {
      if (dom.noActiveAlertsBox) dom.noActiveAlertsBox.style.display = "block";
    } else {
      if (dom.noActiveAlertsBox) dom.noActiveAlertsBox.style.display = "none";
      active.forEach(a => {
        const card = document.createElement("div");
        const sev = (a.severity || "LOW").toUpperCase();
        const stat = (a.status || "ACTIVE").toUpperCase();
        card.className = `alert-item-card alert-card-${sev}`;

        const started = a.started_at ? new Date(a.started_at) : new Date();
        const elapsedMins = Math.max(0, Math.round((Date.now() - started.getTime()) / 60000));
        const elapsedStr = elapsedMins < 60 ? `${elapsedMins} min ago` : `${(elapsedMins / 60).toFixed(1)} h ago`;
        const locStr = [a.district, a.state].filter(Boolean).join(", ") || "Regional";

        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.74rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 5px;">
              ${getHazardIcon(a.hazard)} ${(a.hazard || '').toUpperCase().replace(/_/g, ' ')}
            </span>
            <div style="display: flex; gap: 4px; align-items: center;">
              <span class="badge-pill ${getRiskBadgeClass(sev)}">${sev}</span>
              <span class="alert-lifecycle-badge alert-lifecycle-${stat}">${stat}</span>
            </div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: #94a3b8; margin-top: 3px;">
            <span>📍 ${escapeHtml(locStr)}</span>
            <span>Started ${elapsedStr}</span>
          </div>
          <div style="font-size: 0.67rem; color: #cbd5e1; margin-top: 3px; line-height: 1.25;">
            ${escapeHtml(a.description || a.title || 'Operational hazard active.')}
          </div>
        `;
        dom.activeAlertsList.appendChild(card);
      });
    }
  }

  // Fetch recent lifecycle event history from /api/alerts/history
  fetchAlertHistory();
}

function fetchAlertHistory() {
  fetch("/api/alerts/history?limit=10")
    .then(r => r.json())
    .then(resp => {
      if (!resp.success || !dom.alertHistoryTimeline) return;
      const events = (resp.data && resp.data.events) ? resp.data.events : [];
      if (dom.alertHistoryCount) {
        dom.alertHistoryCount.textContent = `${events.length} events`;
      }
      dom.alertHistoryTimeline.innerHTML = "";
      if (events.length === 0) {
        dom.alertHistoryTimeline.innerHTML = '<div style="color: #64748b; text-align: center; padding: 4px;">No recent alert events logged.</div>';
        return;
      }
      events.forEach(ev => {
        const row = document.createElement("div");
        row.className = "alert-event-row";
        const et = (ev.event_type || "UPDATED").toUpperCase();
        const hz = (ev.hazard || ev.title || "HAZARD").toUpperCase().replace(/_/g, ' ');
        const timeStr = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "--";
        const sevChange = ev.old_severity ? `${ev.old_severity} → ${ev.new_severity}` : ev.new_severity;

        row.innerHTML = `
          <div style="display: flex; align-items: center; gap: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            <span class="alert-lifecycle-badge alert-lifecycle-${et}" style="font-size: 0.58rem; padding: 1px 4px;">${et}</span>
            <span style="font-weight: 600; color: #e2e8f0; font-size: 0.68rem;">${escapeHtml(hz)}:</span>
            <span style="color: #94a3b8; font-size: 0.65rem;">${escapeHtml(sevChange)}</span>
          </div>
          <span style="color: #64748b; font-size: 0.62rem; white-space: nowrap; margin-left: 6px;">${timeStr}</span>
        `;
        dom.alertHistoryTimeline.appendChild(row);
      });
    })
    .catch(() => {});
}

// =============================================================================
// 4. LOADING & EXECUTION CONTROLLERS
// =============================================================================

function startLoading() {
  isAnalyzing = true;
  dom.analyzeBtn.disabled = true;
  dom.btnSpinner.style.display = "inline-block";
  dom.btnText.textContent = "Analyzing...";
  dom.validationMsg.style.display = "none";
  dom.loadingIndicator.style.display = "flex";

  let stageIndex = 0;
  dom.loadingStageText.textContent = LOADING_STAGES[0];
  loadingInterval = setInterval(() => {
    stageIndex = (stageIndex + 1) % LOADING_STAGES.length;
    dom.loadingStageText.textContent = LOADING_STAGES[stageIndex];
  }, 700);
}

function stopLoading() {
  isAnalyzing = false;
  dom.analyzeBtn.disabled = false;
  dom.btnSpinner.style.display = "none";
  dom.btnText.textContent = "Analyze Live Conditions";
  dom.loadingIndicator.style.display = "none";
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
}

function validateInputs() {
  if (!selectedLocation || typeof selectedLocation.lat !== "number" || typeof selectedLocation.lon !== "number" || Number.isNaN(selectedLocation.lat) || Number.isNaN(selectedLocation.lon)) {
    return "Please search and select a location in India before analyzing.";
  }
  if (selectedLocation.lat < -90 || selectedLocation.lat > 90) {
    return "Selected latitude must be between -90 and 90.";
  }
  if (selectedLocation.lon < -180 || selectedLocation.lon > 180) {
    return "Selected longitude must be between -180 and 180.";
  }
  const crop = dom.cropSelect.value ? dom.cropSelect.value.trim() : "";
  const cropStage = dom.cropStageSelect.value ? dom.cropStageSelect.value.trim() : "";

  if (!crop) return "Please select a crop.";
  if (!cropStage) return "Please select a crop stage.";
  return null;
}

// Main Unified Prediction Execution
async function runAnalysis() {
  if (isAnalyzing) return;

  const validationError = validateInputs();
  if (validationError) {
    dom.validationMsg.textContent = validationError;
    dom.validationMsg.style.display = "block";
    return;
  }

  const requestId = ++currentRequestId;
  startLoading();

  // Clear previous location's dynamic overlays immediately so they don't linger
  if (mapState.rainfallLayer) {
    mapState.rainfallLayer.clearLayers();
  }
  if (mapState.satelliteLayer) {
    mapState.satelliteLayer.clearLayers();
  }

  // Internal exact coordinates preserved without rounding
  const payload = {
    lat: selectedLocation.lat,
    lon: selectedLocation.lon,
    crop: dom.cropSelect.value,
    crop_stage: dom.cropStageSelect.value
  };

  try {
    const response = await fetch("/api/predict/all/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (requestId !== currentRequestId) {
      // Stale response: a newer request was dispatched
      return;
    }

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.error || `HTTP ${response.status}: Analysis request failed`);
    }

    const data = await response.json();
    mapState.latestData.core = data;
    renderDashboard(data, requestId);
  } catch (err) {
    if (requestId === currentRequestId) {
      dom.validationMsg.textContent = `Analysis Failed: ${err.message}`;
      dom.validationMsg.style.display = "block";
    }
  } finally {
    if (requestId === currentRequestId) {
      stopLoading();
    }
  }
}

// =============================================================================
// 5. DASHBOARD RENDERER
// =============================================================================

function renderDashboard(data, reqId) {
  if (!data || typeof data !== "object") return;

  const loc = data.location || {};
  const summary = data.summary || {};
  const overallRisk = summary.overall_risk || "--";

  // 1. Update Alerts Bar
  dom.overallRiskBadge.textContent = overallRisk;
  dom.overallRiskBadge.className = `badge-pill ${getRiskBadgeClass(overallRisk)}`;

  const wlRes = (data.waterlogging && data.waterlogging.data && data.waterlogging.data.result) || {};
  if (wlRes.warning === true) {
    dom.alertBarWaterlogging.textContent = `WARNING ACTIVE (${(wlRes.positive_probability * 100).toFixed(1)}% ≥ 0.36)`;
    dom.alertBarWaterlogging.style.color = "#ef4444";
  } else {
    dom.alertBarWaterlogging.textContent = `Normal (${formatVal(wlRes.positive_probability !== undefined ? (wlRes.positive_probability * 100) : null, '%')} < 0.36)`;
    dom.alertBarWaterlogging.style.color = "#10b981";
  }

  const outlook = wlRes.future_outlook || {};
  if (outlook.most_significant_window && outlook.most_significant_window.period) {
    dom.alertBarOutlook.textContent = `${outlook.most_significant_window.period} (~${formatVal(outlook.most_significant_window.total_rain_mm, 'mm')})`;
  } else {
    dom.alertBarOutlook.textContent = outlook.summary || "Nominal forecast";
  }

  const dData = (data.disaster && data.disaster.data) || {};
  const vHazards = dData.verified_hazards || [];

  if (vHazards.length > 0) {
    const hazardTokens = vHazards.map(h => {
      const name = (h.hazard || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
      const sev = (h.severity || "ACTIVE").toUpperCase();
      return `${name} ${sev}`;
    });
    dom.alertBarDisaster.textContent = hazardTokens.join(" · ");
    const maxSev = (vHazards[0].severity || "HIGH").toUpperCase();
    dom.alertBarDisaster.style.color = (maxSev === "CRITICAL" || maxSev === "SEVERE" || maxSev === "HIGH") ? "#ef4444" : "#f59e0b";
  } else {
    dom.alertBarDisaster.textContent = "No Active Disaster Threat";
    dom.alertBarDisaster.style.color = "#10b981";
  }

  const agPred = (data.agriculture && data.agriculture.data && data.agriculture.data.result && data.agriculture.data.result.prediction) || {};
  dom.alertBarAgri.textContent = agPred.crop_weather_risk ? `Risk: ${agPred.crop_weather_risk} • Irr: ${agPred.irrigation || '--'}` : "Nominal";

  // Meta count & latency
  let successCount = 0;
  ["weather", "disaster", "agriculture", "waterlogging"].forEach(mod => {
    if (data[mod] && data[mod].success === true) successCount++;
  });
  dom.summaryModelsCount.textContent = `${successCount} / 4 Successful`;

  const totalTime = data.total_execution_time_ms || (data.execution_times && data.execution_times.total_ms);
  dom.summaryTotalTime.textContent = totalTime ? `${totalTime} ms` : "-- ms";

  const now = new Date();
  const dateStr = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true });
  dom.summaryTimestamp.textContent = `${dateStr}, ${timeStr} IST`;

  // System Advisory
  if (summary.overall_advisory) {
    dom.overallAdvisoryText.textContent = summary.overall_advisory;
    dom.overallAdvisoryBox.style.display = "flex";
  } else {
    dom.overallAdvisoryBox.style.display = "none";
  }

  // 2. Render Side Panels
  renderWeatherPanel(data.weather);
  renderDisasterPanel(data.disaster);
  renderAgriculturePanel(data.agriculture, loc);
  renderWaterloggingPanel(data.waterlogging);

  // 3. Update Interactive Map
  updateSelectedLocationMarker(data);
  renderDisasterEventMarkers(data.disaster);

  // 4. Render Terrain Samples on Map from Phase C data
  const terrainInp = (data.waterlogging && data.waterlogging.data && data.waterlogging.data.input && data.waterlogging.data.input.terrain) || {};
  const enhTerrain = terrainInp.enhanced_terrain || (wlRes.terrain && wlRes.terrain.enhanced_terrain);
  if (enhTerrain && enhTerrain.samples_geojson) {
    renderTerrainSamplePoints(enhTerrain.samples_geojson, enhTerrain.downslope_direction, loc.lat, loc.lon);
  }

  // 5. Asynchronous Non-Blocking Drainage Intelligence Fetch (Phase B)
  fetchDrainageContext(loc.lat, loc.lon, reqId);

  // 6. Asynchronous Non-Blocking Spatial Rainfall Fetch (Phase E Replacement)
  fetchSpatialRainfallContext(loc.lat, loc.lon, reqId);

  // 7. Asynchronous Non-Blocking Sentinel-1 SAR Surface Water Fetch (Phase F)
  fetchSarWaterContext(loc.lat, loc.lon, reqId);

  // 8. Pan-India Hierarchical Spatial Risk Grid Generation (Local District / Area)
  const gridRes = Number((dom.navGridResolution && dom.navGridResolution.value) || 500);
  generateLocalRiskGrid(loc.lat, loc.lon, 5000, gridRes);

  // 9. Telemetry Panels
  renderTelemetry(data);
}


// Weather Panel Renderer
function renderWeatherPanel(w) {
  if (!w || !w.success) {
    dom.weatherStatusBadge.textContent = "ERROR";
    dom.weatherStatusBadge.className = "status-pill status-error";
    dom.weatherErrorBox.textContent = w && w.error ? w.error : "Weather prediction unavailable";
    dom.weatherErrorBox.style.display = "block";
    dom.weatherContent.style.display = "none";
    return;
  }

  dom.weatherStatusBadge.textContent = "SUCCESS";
  dom.weatherStatusBadge.className = "status-pill status-success";
  dom.weatherErrorBox.style.display = "none";
  dom.weatherContent.style.display = "block";

  const feat = (w.data && w.data.input && w.data.input.features) || {};
  const pred = (w.data && w.data.result && w.data.result.prediction) || {};

  dom.wTemp.textContent = formatVal(pred.temperature_2m || feat.temperature_2m, "°C");
  dom.wApparentTemp.textContent = formatVal(pred.apparent_temperature || feat.apparent_temperature, "°C");
  dom.wHumidity.textContent = formatVal(pred.relative_humidity_2m || feat.relative_humidity_2m, "%");
  dom.wDewPoint.textContent = formatVal(pred.dew_point_2m || feat.dew_point_2m, "°C");
  dom.wPrecip.textContent = formatVal(pred.precipitation || feat.precipitation, "mm");
  dom.wPressure.textContent = formatVal(pred.surface_pressure || feat.surface_pressure, "hPa");
  dom.wWindSpeed.textContent = formatVal(pred.wind_speed_10m || feat.wind_speed_10m, "km/h");
  dom.wWindGusts.textContent = formatVal(pred.wind_gusts_10m || feat.wind_gusts_10m, "km/h");
  dom.wCloudCover.textContent = formatVal(pred.cloud_cover || feat.cloud_cover, "%");
  dom.wSoilMoisture.textContent = formatVal(pred.soil_moisture_0_to_7cm || feat.soil_moisture_0_to_7cm, "m³/m³");
  dom.wSoilTemp.textContent = formatVal(pred.soil_temperature_0_to_7cm || feat.soil_temperature_0_to_7cm, "°C");
  dom.wET0.textContent = formatVal(pred.et0_fao_evapotranspiration || feat.et0_fao_evapotranspiration, "mm");
  dom.wRain.textContent = formatVal(pred.rain, "mm");
  dom.wWindDirection.textContent = formatVal(pred.wind_direction_10m, "°");
  dom.wVPD.textContent = formatVal(pred.vapour_pressure_deficit, "kPa");
  dom.wShortwaveRad.textContent = formatVal(pred.shortwave_radiation, "W/m²");
  dom.wDNI.textContent = formatVal(pred.direct_normal_irradiance, "W/m²");
  dom.wDiffuseRad.textContent = formatVal(pred.diffuse_radiation, "W/m²");
}

// Disaster Panel Renderer
function renderDisasterPanel(d) {
  if (!d || !d.success) {
    dom.disasterStatusBadge.textContent = "ERROR";
    dom.disasterStatusBadge.className = "status-pill status-error";
    let errMsg = d && d.error ? d.error : "Disaster hazard inference unavailable";
    dom.disasterErrorBox.textContent = errMsg;
    dom.disasterErrorBox.style.display = "block";
    dom.disasterContent.style.display = "none";
    return;
  }

  dom.disasterStatusBadge.textContent = "SUCCESS";
  dom.disasterStatusBadge.className = "status-pill status-success";
  dom.disasterErrorBox.style.display = "none";
  dom.disasterContent.style.display = "block";

  const dData = d.data || {};
  const res = dData.raw_model || dData.result || {};
  const feat = (dData.input && dData.input.features) || dData.features || {};
  const live = dData.live_evidence || {};
  const hist = dData.historical_context || {};
  const verified = dData.verified_hazard || (dData.result && dData.result.verified_hazard) || null;
  const vHazards = dData.verified_hazards || [];
  const offAlerts = dData.official_alerts || [];

  // Update Data Quality Provenance Badge
  if (dom.disasterQualityBadge) {
    const qual = dData.model_feature_quality || "FULL_DATA";
    dom.disasterQualityBadge.textContent = qual;
    if (qual === "FULL_DATA") dom.disasterQualityBadge.className = "status-pill status-success";
    else if (qual === "PARTIAL_DATA") dom.disasterQualityBadge.className = "status-pill status-warning";
    else if (qual === "FALLBACK_USED") dom.disasterQualityBadge.className = "status-pill status-info";
    else dom.disasterQualityBadge.className = "status-pill status-danger";
  }

  // 1. Verified Operational Hazard Output
  if (verified && dom.dVerifiedBox) {
    const status = verified.status || "NO_ACTIVE_HAZARD";
    if (dom.dVerifiedStatusBadge) {
      dom.dVerifiedStatusBadge.textContent = status;
      if (status === "VERIFIED_ACTIVE_HAZARD") {
        dom.dVerifiedStatusBadge.className = "status-pill status-danger";
        dom.dVerifiedBox.className = "disaster-verified-box verified-active";
      } else if (status === "SUPPORTED_HAZARD_SIGNAL") {
        dom.dVerifiedStatusBadge.className = "status-pill status-warning";
        dom.dVerifiedBox.className = "disaster-verified-box verified-active";
      } else if (status === "UNCORROBORATED_MODEL_SIGNAL") {
        dom.dVerifiedStatusBadge.className = "status-pill status-info";
        dom.dVerifiedBox.className = "disaster-verified-box verified-uncorroborated";
      } else if (status === "NO_ACTIVE_HAZARD") {
        dom.dVerifiedStatusBadge.className = "status-pill status-success";
        dom.dVerifiedBox.className = "disaster-verified-box verified-clear";
      } else {
        dom.dVerifiedStatusBadge.className = "status-pill status-neutral";
        dom.dVerifiedBox.className = "disaster-verified-box";
      }
    }

    if (vHazards.length > 0) {
      dom.dVerifiedStatusBadge.textContent = vHazards.length === 1 ? "VERIFIED HAZARD" : `VERIFIED HAZARDS (${vHazards.length})`;
      dom.dVerifiedStatusBadge.className = "status-pill status-danger";
      dom.dVerifiedBox.className = "disaster-verified-box verified-active";

      if (dom.dVerifiedHazardBadge) {
        const topH = vHazards[0];
        dom.dVerifiedHazardBadge.textContent = `${(topH.hazard || '').toUpperCase().replace(/_/g, ' ')} — ${topH.severity || 'ACTIVE'}`;
        dom.dVerifiedHazardBadge.className = `pred-badge ${getRiskBadgeClass(topH.hazard)}`;
      }
      if (dom.dVerifiedSeverity) {
        dom.dVerifiedSeverity.style.display = "none";
      }

      if (dom.dMultiHazardsContainer) {
        if (vHazards.length > 1) {
          dom.dMultiHazardsContainer.innerHTML = '<span style="font-size:0.68rem; color:#94a3b8; width:100%; margin-bottom:2px; font-weight:600;">Multiple Verified Hazards:</span>' +
            vHazards.map(h => `<span class="badge-pill ${h.severity === 'CRITICAL' || h.severity === 'SEVERE' ? 'badge-danger' : (h.severity === 'HIGH' ? 'badge-warning' : 'badge-info')}" style="font-size:0.75rem; padding: 3px 8px;">${escapeHtml(h.hazard.toUpperCase().replace(/_/g, ' '))} — ${escapeHtml(h.severity || 'ACTIVE')}</span>`).join(' ');
          dom.dMultiHazardsContainer.style.display = "flex";
        } else {
          dom.dMultiHazardsContainer.style.display = "none";
        }
      }
    } else {
      if (dom.dVerifiedHazardBadge) {
        dom.dVerifiedHazardBadge.textContent = "NO ACTIVE LOCAL HAZARD";
        dom.dVerifiedHazardBadge.className = "pred-badge badge-neutral";
      }
      if (dom.dVerifiedSeverity) {
        dom.dVerifiedSeverity.style.display = "inline-block";
        const sev = verified ? (verified.severity || "NONE") : "NONE";
        dom.dVerifiedSeverity.textContent = `SEVERITY: ${sev}`;
        dom.dVerifiedSeverity.className = "badge-pill badge-neutral";
      }
      if (dom.dMultiHazardsContainer) {
        dom.dMultiHazardsContainer.style.display = "none";
      }
    }

    // Official Bulletins & Alerts (Phase 15)
    if (dom.dOfficialAlertsContainer) {
      if (offAlerts.length > 0) {
        dom.dOfficialAlertsContainer.innerHTML = `<strong>Official Bulletins (${offAlerts.length}):</strong> ` +
          offAlerts.map(a => `<div style="margin-top:2px;">• <strong>${escapeHtml(a.source || 'Official Authority')}:</strong> ${escapeHtml(a.title || a.hazard)} [${escapeHtml(a.severity || 'ACTIVE')}] - ${escapeHtml(a.description || '')}</div>`).join('');
        dom.dOfficialAlertsContainer.style.display = "block";
      } else {
        dom.dOfficialAlertsContainer.style.display = "none";
      }
    }

    if (dom.dVerifiedReason) {
      dom.dVerifiedReason.textContent = verified.reason || "--";
    }
  }

  // 1b. Verified Hazard Monitor Cards (Wildfire, Avalanche, Flash Flood)
  const hStatus = dData.hazard_status || {};
  const wf = hStatus.wildfire || {};
  const av = hStatus.avalanche || {};
  const ff = hStatus.flash_flood || {};

  // Wildfire Card
  if (dom.badgeWildfireStatus) {
    if (wf.active) {
      dom.badgeWildfireStatus.textContent = wf.severity || "ACTIVE";
      dom.badgeWildfireStatus.className = `status-pill ${(wf.severity === 'CRITICAL' || wf.severity === 'SEVERE' || wf.severity === 'HIGH') ? 'status-danger' : 'status-warning'}`;
    } else if (wf.confidence_type === "insufficient_data") {
      dom.badgeWildfireStatus.textContent = "UNAVAILABLE";
      dom.badgeWildfireStatus.className = "status-pill status-warning";
    } else {
      dom.badgeWildfireStatus.textContent = "SAFE";
      dom.badgeWildfireStatus.className = "status-pill status-success";
    }
  }
  if (dom.valFireNearest) {
    const wfEv = wf.evidence || {};
    dom.valFireNearest.textContent = wfEv.nearest_fire_distance_km != null ? `${Number(wfEv.nearest_fire_distance_km).toFixed(1)} km` : "None <50km";
  }
  if (dom.valFireCounts) {
    const wfEv = wf.evidence || {};
    dom.valFireCounts.textContent = `${wfEv.detections_10km || 0} / ${wfEv.detections_25km || 0}`;
  }
  if (dom.valFireMaxFrp) {
    const wfEv = wf.evidence || {};
    dom.valFireMaxFrp.textContent = wfEv.max_frp != null ? `${Number(wfEv.max_frp).toFixed(1)} MW` : "0 MW";
  }
  if (dom.descWildfireReason) {
    dom.descWildfireReason.textContent = wf.reason || "No recent thermal anomalies observed.";
  }

  // Avalanche Card
  if (dom.badgeAvalancheStatus) {
    if (av.active) {
      dom.badgeAvalancheStatus.textContent = av.severity || "ACTIVE";
      dom.badgeAvalancheStatus.className = `status-pill ${(av.severity === 'CRITICAL' || av.severity === 'SEVERE' || av.severity === 'HIGH') ? 'status-danger' : 'status-warning'}`;
    } else if (av.confidence_type === "insufficient_data") {
      dom.badgeAvalancheStatus.textContent = "UNAVAILABLE";
      dom.badgeAvalancheStatus.className = "status-pill status-warning";
    } else {
      dom.badgeAvalancheStatus.textContent = "SAFE";
      dom.badgeAvalancheStatus.className = "status-pill status-success";
    }
  }
  if (dom.valAvalancheElev) {
    const avEv = av.evidence || {};
    dom.valAvalancheElev.textContent = avEv.elevation_m != null ? `${Number(avEv.elevation_m).toFixed(0)} m` : "--";
  }
  if (dom.valAvalancheSlope) {
    const avEv = av.evidence || {};
    dom.valAvalancheSlope.textContent = avEv.slope_deg != null ? `${Number(avEv.slope_deg).toFixed(1)}°` : "--";
  }
  if (dom.valAvalancheTemp) {
    const avEv = av.evidence || {};
    dom.valAvalancheTemp.textContent = avEv.temp_2m_c != null ? `${Number(avEv.temp_2m_c).toFixed(1)}°C` : "--";
  }
  if (dom.descAvalancheReason) {
    dom.descAvalancheReason.textContent = av.reason || "No official DGRE/SACHET warning active.";
  }

  // Flash Flood Card
  if (dom.badgeFlashFloodStatus) {
    if (ff.active) {
      dom.badgeFlashFloodStatus.textContent = ff.severity || "ACTIVE";
      dom.badgeFlashFloodStatus.className = `status-pill ${(ff.severity === 'CRITICAL' || ff.severity === 'SEVERE' || ff.severity === 'HIGH') ? 'status-danger' : 'status-warning'}`;
    } else {
      dom.badgeFlashFloodStatus.textContent = "SAFE";
      dom.badgeFlashFloodStatus.className = "status-pill status-success";
    }
  }
  if (dom.valFlashFloodRain) {
    const ffEv = ff.evidence || {};
    dom.valFlashFloodRain.textContent = `${Number(ffEv.rain_1h_mm || 0).toFixed(1)} / ${Number(ffEv.rain_3h_mm || 0).toFixed(1)} mm`;
  }
  if (dom.valFlashFloodProbs) {
    const ffEv = ff.evidence || {};
    const wlP = ffEv.waterlogging_probability != null ? `${(ffEv.waterlogging_probability * 100).toFixed(0)}%` : "--";
    const drP = ffEv.drainage_congestion_probability != null ? `${(ffEv.drainage_congestion_probability * 100).toFixed(0)}%` : "--";
    dom.valFlashFloodProbs.textContent = `${wlP} / ${drP}`;
  }
  if (dom.valFlashFloodSar) {
    const ffEv = ff.evidence || {};
    dom.valFlashFloodSar.textContent = ffEv.surface_water_change ? "Change Detected" : "No Change";
  }
  if (dom.descFlashFloodReason) {
    dom.descFlashFloodReason.textContent = ff.reason || "Short-duration precipitation nominal.";
  }

  // Operational Active Alerts and Event History (Section 15)
  renderActiveAlertsAndHistory(dData.alerts || {}, dData.location || selectedLocation);

  // 2. Raw ML Signal (XGBoost Classifier)
  const dClass = res.prediction || "--";
  if (dom.dClass) {
    dom.dClass.textContent = dClass;
    dom.dClass.className = `pred-badge ${getRiskBadgeClass(dClass)}`;
  }

  const conf = res.confidence !== undefined ? `${(res.confidence * 100).toFixed(1)}%` : "--";
  if (dom.dConfidence) {
    dom.dConfidence.textContent = conf;
  }

  // Raw ML Class Probabilities (Renamed UI contract)
  if (res.probabilities && typeof res.probabilities === "object") {
    if (dom.dProbList) {
      dom.dProbList.innerHTML = "";
      const entries = Object.entries(res.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

      entries.forEach(([cls, prob]) => {
        const chip = document.createElement("span");
        chip.className = "prob-chip";
        chip.textContent = `${cls}: ${(prob * 100).toFixed(1)}%`;
        dom.dProbList.appendChild(chip);
      });
    }
    if (dom.disasterProbabilitiesBox) {
      dom.disasterProbabilitiesBox.style.display = "block";
    }
  } else if (dom.disasterProbabilitiesBox) {
    dom.disasterProbabilitiesBox.style.display = "none";
  }

  // 3. Live Telemetry & Seismic Evidence
  if (dom.dNearestQuakeDist) {
    if (live.nearest_quake_km != null) {
      dom.dNearestQuakeDist.textContent = `${Number(live.nearest_quake_km).toFixed(0)} km`;
    } else {
      dom.dNearestQuakeDist.textContent = "None <500km";
    }
  }
  if (dom.dNearestQuakeAge) {
    if (live.nearest_quake_age_days != null) {
      dom.dNearestQuakeAge.textContent = `${Number(live.nearest_quake_age_days).toFixed(1)} d`;
    } else {
      dom.dNearestQuakeAge.textContent = "--";
    }
  }
  if (dom.dMaxQuakeMag) {
    const mq = live.max_quake_mag != null ? live.max_quake_mag : feat.max_quake_mag;
    dom.dMaxQuakeMag.textContent = mq != null && mq > 0 ? `M${Number(mq).toFixed(1)}` : "0.0";
  }
  if (dom.dSeismicStatus) {
    dom.dSeismicStatus.textContent = live.seismic_status || "Not locally active";
  }
  if (dom.dRiverDischarge) {
    const rd = live.river_discharge != null ? live.river_discharge : feat.river_discharge;
    dom.dRiverDischarge.textContent = formatVal(rd, "m³/s");
  }
  if (dom.dTsunamiFlag) {
    const tf = live.tsunami_flag != null ? live.tsunami_flag : feat.tsunami_flag;
    dom.dTsunamiFlag.textContent = tf === 1 ? "Active (1)" : "None (0)";
  }
  if (dom.dCycloneFlag) {
    const cf = live.gdacs_cyclone != null ? live.gdacs_cyclone : feat.gdacs_cyclone;
    dom.dCycloneFlag.textContent = (cf === 1 || cf === true) ? "Alert Active (1)" : "None (0)";
  }

  // 4. Historical & Recent Multi-Window Context
  const eqHist = hist.earthquake || {};
  const fHist = hist.flood || {};
  const rHist = hist.rainfall || {};
  const drHist = hist.drought || {};

  if (dom.dQuakes24h) dom.dQuakes24h.textContent = eqHist.events_24h != null ? eqHist.events_24h : "0";
  if (dom.dQuakes7d) dom.dQuakes7d.textContent = eqHist.events_7d != null ? eqHist.events_7d : "0";
  if (dom.dQuakes30d) dom.dQuakes30d.textContent = eqHist.events_30d != null ? eqHist.events_30d : "0";
  if (dom.dQuakes90d) dom.dQuakes90d.textContent = eqHist.events_90d != null ? eqHist.events_90d : "0";

  if (dom.dPeakDischarge7d) {
    const pd = fHist.peak_discharge_7d != null ? fHist.peak_discharge_7d : fHist.peak_discharge;
    dom.dPeakDischarge7d.textContent = pd != null ? `${Number(pd).toFixed(1)} m³/s` : "--";
  }
  if (dom.dDischargeTrend) {
    dom.dDischargeTrend.textContent = fHist.discharge_trend || "--";
  }
  if (dom.dRain7d) {
    dom.dRain7d.textContent = rHist.accumulation_7d_mm != null ? `${Number(rHist.accumulation_7d_mm).toFixed(1)} mm` : "--";
  }
  if (dom.dDroughtAnomaly) {
    dom.dDroughtAnomaly.textContent = drHist.status || drHist.soil_status || "Normal";
  }

  // 5. Engineered Indicators
  if (dom.dShear) dom.dShear.textContent = formatVal(feat.wind_pressure_shear);
  if (dom.dSaturation) dom.dSaturation.textContent = formatVal(feat.precipitation_saturation_rate);
  if (dom.dThermal) dom.dThermal.textContent = formatVal(feat.thermal_particulate_index);
}

// Agriculture Panel Renderer
function renderAgriculturePanel(ag, loc) {
  if (!ag || !ag.success) {
    dom.agStatusBadge.textContent = "ERROR";
    dom.agStatusBadge.className = "status-pill status-error";
    dom.agErrorBox.textContent = ag && ag.error ? ag.error : "Agriculture advisory unavailable";
    dom.agErrorBox.style.display = "block";
    dom.agContent.style.display = "none";
    return;
  }

  dom.agStatusBadge.textContent = "SUCCESS";
  dom.agStatusBadge.className = "status-pill status-success";
  dom.agErrorBox.style.display = "none";
  dom.agContent.style.display = "block";

  const agData = ag.data || {};
  const pred = (agData.result && agData.result.prediction) || {};

  dom.agCropDisplay.textContent = agData.crop || loc.crop || "--";
  dom.agStageDisplay.textContent = agData.crop_stage || loc.crop_stage || "--";

  const targetMap = [
    { el: dom.agTargetWeatherRisk, val: pred.crop_weather_risk },
    { el: dom.agTargetIrrigation, val: pred.irrigation },
    { el: dom.agTargetHeatStress, val: pred.heat_stress },
    { el: dom.agTargetWaterlogging, val: pred.waterlogging_risk },
    { el: dom.agTargetSpraying, val: pred.spraying_suitability },
  ];

  targetMap.forEach(({ el, val }) => {
    const text = val || "--";
    el.textContent = text;
    el.className = `badge-pill ${getRiskBadgeClass(text)}`;
  });

  const probs = agData.result && agData.result.probabilities;
  if (probs && typeof probs === "object") {
    dom.agProbList.innerHTML = "";
    Object.entries(probs).forEach(([target, classProbs]) => {
      if (typeof classProbs === "object") {
        Object.entries(classProbs).forEach(([cName, pVal]) => {
          const chip = document.createElement("span");
          chip.className = "prob-chip";
          chip.textContent = `${target} [${cName}]: ${(pVal * 100).toFixed(1)}%`;
          dom.agProbList.appendChild(chip);
        });
      }
    });
    dom.agProbabilitiesBox.style.display = dom.agProbList.children.length ? "block" : "none";
  } else {
    dom.agProbabilitiesBox.style.display = "none";
  }
}

// Waterlogging Panel Renderer
function renderWaterloggingPanel(wl) {
  if (!wl || !wl.success) {
    dom.wlStatusBadge.textContent = "ERROR";
    dom.wlStatusBadge.className = "status-pill status-error";
    dom.wlErrorBox.textContent = wl && wl.error ? wl.error : "Waterlogging hazard prediction unavailable";
    dom.wlErrorBox.style.display = "block";
    dom.wlContent.style.display = "none";
    return;
  }

  dom.wlStatusBadge.textContent = "SUCCESS";
  dom.wlStatusBadge.className = "status-pill status-success";
  dom.wlErrorBox.style.display = "none";
  dom.wlContent.style.display = "block";

  const res = (wl.data && wl.data.result) || {};
  const inp = (wl.data && wl.data.input) || {};

  // Warning Banner
  if (res.warning === true) {
    dom.wlWarningBanner.style.display = "flex";
  } else {
    dom.wlWarningBanner.style.display = "none";
  }

  const risk = res.risk_level || "--";
  dom.wlRiskLevel.textContent = risk;
  dom.wlRiskLevel.className = `pred-badge ${getRiskBadgeClass(risk)}`;

  const posProb = res.positive_probability !== undefined ? `${(res.positive_probability * 100).toFixed(1)}%` : "--";
  dom.wlProb.textContent = posProb;

  // Recent Rainfall Windows
  const rain = inp.rainfall_summary_mm || {};
  const recentMap = res.recent_rainfall || inp.recent_rainfall || {};

  dom.wlRain1h.textContent = formatVal(rain.rainfall_1h, "mm");
  dom.wlRain3h.textContent = formatVal(rain.rainfall_3h !== undefined ? rain.rainfall_3h : (recentMap["3h"] && recentMap["3h"].value), "mm");
  dom.wlRain6h.textContent = formatVal(rain.rainfall_6h !== undefined ? rain.rainfall_6h : (recentMap["6h"] && recentMap["6h"].value), "mm");
  dom.wlRain12h.textContent = formatVal(rain.rainfall_12h !== undefined ? rain.rainfall_12h : (recentMap["12h"] && recentMap["12h"].value), "mm");
  dom.wlRain24h.textContent = formatVal(rain.rainfall_24h, "mm");
  dom.wlRain3d.textContent = formatVal(rain.rainfall_3d, "mm");
  dom.wlRain7d.textContent = formatVal(rain.rainfall_7d, "mm");

  dom.wlTimelinePast24h.textContent = formatVal(rain.rainfall_24h, "mm");
  dom.wlTimelinePast7d.textContent = formatVal(rain.rainfall_7d, "mm");

  // Forecast Rainfall Windows
  const fc = res.forecast_rainfall || inp.forecast_rainfall || {};
  dom.wlFcNext1h.textContent = formatVal(fc.next_1h && fc.next_1h.value, "mm");
  dom.wlFcNext3h.textContent = formatVal(fc.next_3h && fc.next_3h.value, "mm");
  dom.wlFcNext6h.textContent = formatVal(fc.next_6h && fc.next_6h.value, "mm");
  dom.wlFcNext12h.textContent = formatVal(fc.next_12h && fc.next_12h.value, "mm");
  dom.wlFcNext24h.textContent = formatVal(fc.next_24h && fc.next_24h.value, "mm");
  dom.wlFcNext48h.textContent = formatVal(fc.next_48h && fc.next_48h.value, "mm");
  dom.wlFcNext72h.textContent = formatVal(fc.next_72h && fc.next_72h.value, "mm");

  dom.wlTimelineFuture24h.textContent = formatVal(fc.next_24h && fc.next_24h.value, "mm");
  dom.wlTimelineFuture72h.textContent = formatVal(fc.next_72h && fc.next_72h.value, "mm");

  // Terrain Intelligence
  const terrain = inp.terrain || res.terrain || {};
  dom.wlElevation.textContent = formatVal(terrain.elevation_m, "m");
  dom.wlSlope.textContent = formatVal(terrain.slope_deg, "°");

  const enh = terrain.enhanced_terrain;
  if (enh) {
    dom.wlTerrainStatus.textContent = enh.status === "ok" ? "READY" : (enh.status === "insufficient_context_samples" ? "PARTIAL" : "UNAVAILABLE");
    dom.wlTerrainStatus.className = `status-pill ${enh.status === "ok" ? "status-success" : "status-neutral"}`;
    dom.wlRelativeElevation.textContent = formatVal(enh.relative_elevation_m, "m");
    dom.wlLocalRelief.textContent = formatVal(enh.local_relief_m, "m");

    if (enh.gradient_m_per_km !== null && enh.gradient_m_per_km !== undefined) {
      dom.wlGradient.textContent = `${formatVal(enh.gradient_m_per_km, "m/km")}`;
    } else {
      dom.wlGradient.textContent = "--";
    }

    if (enh.downslope_direction) {
      const d = enh.downslope_direction;
      dom.wlDownslopeDirection.textContent = d.cardinal === "FLAT" ? "Flat / < 0.5 m/km" : `${d.cardinal} ${d.arrow || ''}`;
    } else {
      dom.wlDownslopeDirection.textContent = "--";
    }

    const ns = enh.profiles && enh.profiles.north_south;
    if (ns && ns.length === 3) {
      dom.wlProfileNS.textContent = `S: ${ns[0].elevation_m}m → C: ${ns[1].elevation_m}m → N: ${ns[2].elevation_m}m`;
    } else {
      dom.wlProfileNS.textContent = "--";
    }

    const ew = enh.profiles && enh.profiles.east_west;
    if (ew && ew.length === 3) {
      dom.wlProfileEW.textContent = `W: ${ew[0].elevation_m}m → C: ${ew[1].elevation_m}m → E: ${ew[2].elevation_m}m`;
    } else {
      dom.wlProfileEW.textContent = "--";
    }
  }

  // Future Outlook
  const outlook = res.future_outlook || inp.future_outlook || {};
  dom.wlOutlookSummary.textContent = outlook.summary || "Future rainfall forecast unavailable";
  dom.wlOutlookBadge.textContent = outlook.severity || "--";
  dom.wlOutlookBadge.className = `badge-pill ${getRiskBadgeClass(outlook.severity)}`;
  dom.wlMaxHourlyRain.textContent = formatVal(outlook.max_hourly_rain_mm, "mm/h");
  dom.wlTotalForecast72h.textContent = formatVal(outlook.total_cumulative_72h_mm, "mm");
  if (outlook.most_significant_window && outlook.most_significant_window.period) {
    dom.wlMostSignificantPeriod.textContent = `${outlook.most_significant_window.period} (~${formatVal(outlook.most_significant_window.total_rain_mm, "mm")})`;
  } else {
    dom.wlMostSignificantPeriod.textContent = "No heavy window forecast";
  }
}

// Asynchronous Non-Blocking Drainage Intelligence Fetcher (Phase B)
async function fetchDrainageContext(lat, lon, reqId) {
  if (!dom.wlDrainageBox) return;

  dom.wlDrainStatus.textContent = "LOADING...";
  dom.wlDrainStatus.className = "status-pill status-neutral";
  dom.wlDrainDistance.textContent = "Querying...";
  dom.wlDrainName.textContent = "--";
  dom.wlDrainType.textContent = "--";
  dom.wlDrainCount.textContent = "--";
  dom.wlDrainLength.textContent = "--";
  dom.wlDrainDensity.textContent = "--";
  dom.wlDrainProximityBadge.textContent = "--";
  dom.wlDrainProximityBadge.className = "badge-pill badge-neutral";

  try {
    const url = `/api/waterlogging/drainage?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_m=2000`;
    const res = await fetch(url);
    if (reqId !== currentRequestId) return; // Stale request protection

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const d = json.data || json;
    mapState.latestData.drainage = d;

    renderDrainageSubsection(d);
    renderDrainageGeoJSON(d, lat, lon);
    renderSentinel2MapMarker(d, lat, lon);
  } catch (err) {
    if (reqId !== currentRequestId) return;
    dom.wlDrainStatus.textContent = "UNAVAILABLE";
    dom.wlDrainStatus.className = "status-pill status-high";
    dom.wlDrainDistance.textContent = "Unavailable";
    dom.wlDrainSourceNote.textContent = `Source: OpenStreetMap via Overpass API | Service temporarily unavailable (${err.message}). Core predictions remain active.`;
    renderDrainageGeoJSON({ status: "unavailable" }, lat, lon);
  }
}

// Render Sentinel-2 Observation Marker / Footprint on Map
function renderSentinel2MapMarker(d, lat, lon) {
  if (!mapState.sentinel2Layer) return;
  mapState.sentinel2Layer.clearLayers();
  const s2 = d.sentinel2_metadata || (d.satellite ? d.satellite.sentinel2 : null);
  if (!s2 || s2.status !== "available") return;
  mapState.latestData.sentinel2 = s2;

  const dtStr = s2.scene_date ? s2.scene_date.split("T")[0] : "Recent";
  const bu = s2.bands_used || {};
  const buHtml = bu.B04 !== undefined ? `<b>Bands Used:</b> B04:${bu.B04}, B08:${bu.B08}, B11:${bu.B11}<br/>` : "";
  const popupHtml = `
    <div style="font-size: 0.85rem; line-height: 1.4; min-width: 220px;">
      <strong style="color: #2b6cb0;">Sentinel-2 L2A Optical Observation</strong><br/>
      <span class="badge-pill badge-safe" style="font-size: 0.65rem; padding: 1px 5px;">LIVE SURFACE</span><br/>
      <b>Scene ID:</b> <span style="font-size:0.75rem;">${escapeHtml(s2.scene_id || '--')}</span><br/>
      <b>Date:</b> ${escapeHtml(dtStr)}<br/>
      <b>Cloud Cover:</b> ${s2.cloud_cover !== undefined ? s2.cloud_cover + '%' : '--'}<br/>
      ${buHtml}
      <b>NDVI (Vegetation):</b> <strong>${s2.ndvi !== undefined ? s2.ndvi : '--'}</strong><br/>
      <b>NDBI (Built-up):</b> <strong>${s2.ndbi !== undefined ? s2.ndbi : '--'}</strong><br/>
      <b>Sampling:</b> ${escapeHtml(s2.sampling_mode || 'exact_pixel')}<br/>
      <b>Provider:</b> ${escapeHtml(s2.provider || 'STAC')}
    </div>
  `;

  const circle = L.circle([lat, lon], {
    radius: 350,
    color: "#2b6cb0",
    fillColor: "#4299e1",
    fillOpacity: 0.18,
    weight: 2,
    dashArray: "4, 4"
  }).bindPopup(popupHtml);

  mapState.sentinel2Layer.addLayer(circle);
}

// Render Sentinel-1 SAR Observation Marker / Footprint on Map (Phase F / Step 10)
function renderSentinel1MapMarker(d, lat, lon) {
  if (!mapState.map || !mapState.satelliteLayer) return;
  const s1 = d.sentinel1_metadata || (d.satellite ? d.satellite.sentinel1 : null) || (d.provenance ? d : null);
  if (!s1) return;
  mapState.latestData.satellite_s1 = s1;

  const prov = s1.provenance && s1.provenance.sentinel1_vv ? s1.provenance.sentinel1_vv : {};
  const isLive = s1.status === "available" || prov.quality === "LIVE";
  const sceneId = s1.scene_id || prov.scene_id || "Regional Baseline";
  const sceneDate = s1.scene_date || prov.scene_date || "Historical";
  const dtStr = sceneDate && sceneDate.includes("T") ? sceneDate.split("T")[0] : sceneDate;
  const vvDb = s1.vv_db !== undefined ? s1.vv_db : (prov.value !== undefined ? prov.value : -10.5);
  const orbit = s1.orbit_direction || prov.orbit_direction || "N/A";
  const sampling = s1.sampling_mode || prov.sampling_mode || "exact_pixel";
  const provider = s1.provider || "Microsoft Planetary Computer";

  const popupHtml = `
    <div style="font-size: 0.85rem; line-height: 1.4; min-width: 230px;">
      <strong style="color: #0284c7;">Sentinel-1 SAR Observation Footprint</strong><br/>
      <span class="badge-pill ${isLive ? 'badge-safe' : 'badge-neutral'}" style="font-size: 0.65rem; padding: 1px 5px;">${isLive ? 'LIVE RADAR' : 'FALLBACK'}</span><br/>
      <b>Scene ID:</b> <span style="font-size:0.72rem; word-break: break-all;">${escapeHtml(sceneId)}</span><br/>
      <b>Scene Date:</b> ${escapeHtml(dtStr)}<br/>
      <b>VV dB:</b> <strong>${Number(vvDb).toFixed(2)} dB</strong><br/>
      <b>Orbit:</b> ${escapeHtml(orbit)}<br/>
      <b>Sampling Mode:</b> ${escapeHtml(sampling)}<br/>
      <b>Provider:</b> ${escapeHtml(provider)}
    </div>
  `;

  // Draw observation ROI circle (500m radius representing SAR sample footprint)
  const circle = L.circle([lat, lon], {
    radius: 500,
    color: isLive ? "#0284c7" : "#64748b",
    fillColor: isLive ? "#38bdf8" : "#94a3b8",
    fillOpacity: 0.22,
    weight: 2,
    dashArray: isLive ? "6, 4" : "2, 4"
  }).bindPopup(popupHtml);

  mapState.satelliteLayer.addLayer(circle);
}

// Render Drainage Panel Subsection
function renderDrainageSubsection(d) {
  if (!d) return;

  const fp = d.feature_provenance || {};

  // Render Data Quality Badge and Feature Provenance
  if (dom.wlDrainDataQualityBadge) {
    const dq = d.data_quality;
    if (dq) {
      const st = (dq.status || "PARTIAL_LIVE").toUpperCase();
      if (st === "FULL_LIVE") {
        dom.wlDrainDataQualityBadge.textContent = "FULL LIVE";
        dom.wlDrainDataQualityBadge.className = "badge-quality-live";
      } else if (st === "FALLBACK_HEAVY") {
        dom.wlDrainDataQualityBadge.textContent = "FALLBACK DATA";
        dom.wlDrainDataQualityBadge.className = "badge-quality-fallback";
      } else {
        dom.wlDrainDataQualityBadge.textContent = "PARTIAL LIVE";
        dom.wlDrainDataQualityBadge.className = "badge-quality-partial";
      }

      // Build provenance tooltip
      const elevSource = fp.elevation ? `${fp.elevation.source} [${fp.elevation.quality}]` : "Open-Meteo DEM (Live)";
      const slopeSource = fp.slope ? `${fp.slope.source} [${fp.slope.quality}]` : "DEM Gradient (Derived)";
      const rainSource = fp.rain_max ? `${fp.rain_max.source} [${fp.rain_max.quality}]` : "Open-Meteo (Derived)";
      const s1Source = fp.sentinel1_vv ? `${fp.sentinel1_vv.source} [${fp.sentinel1_vv.quality}]` : "Baseline (Fallback)";
      const ndviSource = fp.NDVI ? `${fp.NDVI.source} [${fp.NDVI.quality}]` : "Regional Baseline (Fallback)";
      const ndbiSource = fp.NDBI ? `${fp.NDBI.source} [${fp.NDBI.quality}]` : "Regional Baseline (Fallback)";

      const tooltipText = `Elevation: ${elevSource} | Slope: ${slopeSource} | Rain: ${rainSource} | Sentinel-1: ${s1Source} | NDVI: ${ndviSource} | NDBI: ${ndbiSource}`;
      dom.wlDrainDataQualityBadge.title = tooltipText;
    }
  }

  // Render Sentinel-2 Optical Values (NDVI & NDBI) in Drainage Panel
  if (dom.wlDrainNdviVal && dom.wlDrainNdbiVal) {
    const ndviMeta = fp.NDVI || {};
    const ndbiMeta = fp.NDBI || {};
    const ndviVal = ndviMeta.value !== undefined ? ndviMeta.value : (d.model_features ? d.model_features.NDVI : "--");
    const ndbiVal = ndbiMeta.value !== undefined ? ndbiMeta.value : (d.model_features ? d.model_features.NDBI : "--");

    dom.wlDrainNdviVal.textContent = typeof ndviVal === "number" ? ndviVal.toFixed(3) : ndviVal;
    dom.wlDrainNdbiVal.textContent = typeof ndbiVal === "number" ? ndbiVal.toFixed(3) : ndbiVal;

    const isLive = ndviMeta.quality === "LIVE";
    if (dom.wlDrainOpticalBadge) {
      dom.wlDrainOpticalBadge.textContent = isLive ? "LIVE S2" : "BASELINE";
      dom.wlDrainOpticalBadge.className = `badge-pill ${isLive ? "badge-safe" : "badge-neutral"}`;
      dom.wlDrainOpticalBadge.title = ndviMeta.source || "Sentinel-2 L2A";
    }

    if (dom.wlDrainOpticalDate) {
      if (ndviMeta.scene_date) {
        try {
          const dt = ndviMeta.scene_date.split("T")[0];
          dom.wlDrainOpticalDate.textContent = `(${dt})`;
        } catch (_) {
          dom.wlDrainOpticalDate.textContent = "";
        }
      } else {
        dom.wlDrainOpticalDate.textContent = "(baseline)";
      }
    }
  }

  // Render Sentinel-1 SAR Values (VV Backscatter dB) in Drainage Panel
  if (dom.wlDrainSarVal) {
    const s1Meta = fp.sentinel1_vv || {};
    const s1Raw = d.sentinel1_metadata || (d.satellite ? d.satellite.sentinel1 : null);
    const s1Val = s1Meta.value !== undefined ? s1Meta.value : (d.model_features ? d.model_features.sentinel1_vv : "--");

    dom.wlDrainSarVal.textContent = typeof s1Val === "number" ? `${s1Val.toFixed(2)} dB` : s1Val;

    const isLiveS1 = s1Meta.quality === "LIVE";
    if (dom.wlDrainSarBadge) {
      if (isLiveS1) {
        dom.wlDrainSarBadge.textContent = "LIVE S1";
        dom.wlDrainSarBadge.className = "badge-pill badge-safe";
        dom.wlDrainSarBadge.title = s1Meta.source || "Sentinel-1 RTC";
      } else if (s1Raw && s1Raw.status === "no_recent_scene") {
        dom.wlDrainSarBadge.textContent = "NO RECENT SCENE";
        dom.wlDrainSarBadge.className = "badge-pill badge-warning";
        dom.wlDrainSarBadge.title = "No recent scene; using baseline";
      } else {
        dom.wlDrainSarBadge.textContent = "FALLBACK";
        dom.wlDrainSarBadge.className = "badge-pill badge-neutral";
        dom.wlDrainSarBadge.title = "Calibrated regional baseline (-10.5 dB)";
      }
    }

    if (dom.wlDrainSarDate) {
      if (s1Meta.scene_date) {
        try {
          const dt = s1Meta.scene_date.split("T")[0];
          dom.wlDrainSarDate.textContent = `(${dt})`;
        } catch (_) {
          dom.wlDrainSarDate.textContent = "";
        }
      } else {
        dom.wlDrainSarDate.textContent = "(baseline)";
      }
    }

    // Enable satellite layer toggle if Sentinel-1 live observation exists
    if (s1Raw && (s1Raw.status === "available" || isLiveS1)) {
      if (dom.layerItemSatellite && dom.layerBadgeSatellite && dom.layerToggleSatellite) {
        dom.layerItemSatellite.className = "layer-toggle-item active-layer";
        dom.layerBadgeSatellite.textContent = "ACTIVE";
        dom.layerBadgeSatellite.className = "layer-badge-active";
        dom.layerToggleSatellite.disabled = false;
      }
    }

    // Render footprint on map if selectedLocation is present
    if (selectedLocation && (s1Raw || isLiveS1)) {
      renderSentinel1MapMarker(d, selectedLocation.lat, selectedLocation.lon);
    }
  }

  // Render Sentinel-2 status in Section 7 and map toggle
  const s2 = d.sentinel2_metadata || (d.satellite ? d.satellite.sentinel2 : null);
  if (dom.wlS2StatusPill) {
    if (s2 && s2.status === "available") {
      dom.wlS2StatusPill.textContent = "AVAILABLE";
      dom.wlS2StatusPill.className = "status-pill status-safe";
      const dtStr = s2.scene_date ? s2.scene_date.split("T")[0] : "";
      if (dom.wlS2SceneInfo) {
        dom.wlS2SceneInfo.textContent = `${s2.scene_id || 'S2 Scene'} (${dtStr}, Cloud: ${s2.cloud_cover}%)`;
      }
      if (dom.layerItemSentinel2 && dom.layerBadgeSentinel2 && dom.layerToggleSentinel2) {
        dom.layerItemSentinel2.className = "layer-toggle-item active-layer";
        dom.layerBadgeSentinel2.textContent = "ACTIVE";
        dom.layerBadgeSentinel2.className = "layer-badge-active";
        dom.layerToggleSentinel2.disabled = false;
      }
    } else if (s2 && s2.status === "no_recent_scene") {
      dom.wlS2StatusPill.textContent = "NO RECENT SCENE";
      dom.wlS2StatusPill.className = "status-pill status-warning";
      if (dom.wlS2SceneInfo) dom.wlS2SceneInfo.textContent = "No cloud-free scene < 60% in 90 days";
    } else {
      dom.wlS2StatusPill.textContent = "FALLBACK BASELINE";
      dom.wlS2StatusPill.className = "status-pill status-neutral";
      if (dom.wlS2SceneInfo) dom.wlS2SceneInfo.textContent = "Calibrated regional agro-climatic baseline";
    }
  }

  // Render trained Random Forest drainage model prediction if available
  const mp = d.model_prediction || d.drainage_prediction;
  if (mp && dom.wlDrainModelRisk && dom.wlDrainModelProb) {
    const risk = (mp.risk_level || (mp.prediction === 1 ? "HIGH" : "LOW")).toUpperCase();
    dom.wlDrainModelRisk.textContent = risk;
    dom.wlDrainModelRisk.className = `badge-pill ${mp.badge_class || (risk === "HIGH" ? "badge-danger" : risk === "MEDIUM" ? "badge-warning" : "badge-safe")}`;
    const pVal = mp.positive_probability !== undefined ? mp.positive_probability : (mp.probabilities ? mp.probabilities[1] : null);
    dom.wlDrainModelProb.textContent = pVal !== null ? `${(pVal * 100).toFixed(1)}%` : "--";

    if (dom.wlDrainFeatureSummary && d.model_features) {
      const mf = d.model_features;
      const rf = d.rainfall_provenance || {};
      const rawMm = rf.raw_rain_max_mm !== undefined ? rf.raw_rain_max_mm : (d.raw_rain_max_mm !== undefined ? d.raw_rain_max_mm : null);
      const rainLabel = rawMm !== null ? `${mf.rain_max} (${rawMm}mm)` : `${mf.rain_max}`;
      dom.wlDrainFeatureSummary.textContent = `Elev: ${mf.elevation}m | Slope: ${mf.slope}° | Rain: ${rainLabel} | S1: ${mf.sentinel1_vv}dB | NDVI: ${mf.NDVI} | NDBI: ${mf.NDBI}`;
    }
  }

  if (d.status === "no_mapped_data_in_radius") {
    dom.wlDrainStatus.textContent = "NO MAPPED OSM WATERWAYS";
    dom.wlDrainStatus.className = "status-pill status-neutral";
    dom.wlDrainDistance.textContent = "Not mapped within 2 km";
    dom.wlDrainName.textContent = "None mapped";
    dom.wlDrainType.textContent = "None";
    dom.wlDrainCount.textContent = "0";
    dom.wlDrainLength.textContent = "0.00 km";
    dom.wlDrainDensity.textContent = "0.0000 km/km²";
    dom.wlDrainProximityBadge.textContent = "NO MAPPED OSM WATERWAYS IN RADIUS";
    dom.wlDrainProximityBadge.className = "badge-pill badge-neutral";
    dom.wlDrainSourceNote.textContent = d.note || "Source: Random Forest (6-feat) & OpenStreetMap via Overpass API";
    return;
  }

  if (d.status === "unavailable") {
    dom.wlDrainStatus.textContent = "UNAVAILABLE";
    dom.wlDrainStatus.className = "status-pill status-high";
    dom.wlDrainDistance.textContent = "—";
    dom.wlDrainName.textContent = "—";
    dom.wlDrainType.textContent = "—";
    dom.wlDrainCount.textContent = "—";
    dom.wlDrainLength.textContent = "—";
    dom.wlDrainDensity.textContent = "—";
    dom.wlDrainProximityBadge.textContent = "DRAINAGE DATA UNAVAILABLE";
    dom.wlDrainProximityBadge.className = "badge-pill badge-warning";
    dom.wlDrainSourceNote.textContent = d.note || "Source: Random Forest (6-feat) & OpenStreetMap";
    return;
  }

  dom.wlDrainStatus.textContent = d.cache_hit ? "CACHED" : "LIVE";
  dom.wlDrainStatus.className = "status-pill status-safe";

  const distM = d.nearest_drain_distance_m;
  dom.wlDrainDistance.textContent = (distM !== null && distM !== undefined) ? `${distM.toFixed(1)} m` : "--";
  dom.wlDrainName.textContent = d.nearest_drain_name || "Unnamed";
  dom.wlDrainType.textContent = d.nearest_drain_type ? d.nearest_drain_type.toUpperCase() : "--";
  dom.wlDrainCount.textContent = formatVal(d.waterway_count, "", 0);
  dom.wlDrainLength.textContent = formatVal(d.total_waterway_length_km, "km", 2);
  dom.wlDrainDensity.textContent = formatVal(d.drainage_density_km_per_km2, "km/km²", 4);

  const prox = d.proximity_advisory || {};
  dom.wlDrainProximityBadge.textContent = prox.badge_text || "--";
  dom.wlDrainProximityBadge.className = `badge-pill ${prox.badge_class || "badge-neutral"}`;
  dom.wlDrainSourceNote.textContent = d.note || "Source: Random Forest (6-feat) & OpenStreetMap (2km buffer).";
}

// Asynchronous Non-Blocking Open-Meteo Spatial Rainfall Fetcher (Phase E Replacement)
async function fetchSpatialRainfallContext(lat, lon, reqId) {
  if (!dom.wlSpatialSection) return;

  dom.wlSpatialStatus.textContent = "QUERYING...";
  dom.wlSpatialStatus.className = "status-pill status-neutral";
  dom.wlSpatialTime.textContent = "Querying Open-Meteo Recent Model Data...";
  dom.wlSpatialGridInfo.textContent = "Spatial Rainfall Sampling Grid — 0.1\u00b0 request spacing (11 \u00d7 11)";
  dom.wlSpatial1h.textContent = "--";
  dom.wlSpatial3h.textContent = "--";
  dom.wlSpatial6h.textContent = "--";
  dom.wlSpatial12h.textContent = "--";
  dom.wlSpatial24h.textContent = "--";
  dom.wlSpatial3d.textContent = "--";
  dom.wlSpatial7d.textContent = "--";

  try {
    const url = `/api/waterlogging/rainfall/spatial?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&delta_deg=0.5`;
    const res = await fetch(url);
    if (reqId !== currentRequestId) return; // Stale request protection

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const d = json.data || json;
    mapState.latestData.rainfall = d;

    renderSpatialSubsection(d);
    renderSpatialWorkflowCard(d);

    if (d.status === "available" && d.geojson) {
      dom.layerToggleRainfall.disabled = false;
      dom.layerItemRainfall.className = "layer-toggle-item active-layer";
      dom.layerBadgeRainfall.className = "layer-badge-active";
      dom.layerBadgeRainfall.textContent = "AVAILABLE";

      if (mapState.activeLayers.rainfall) {
        renderRainfallGeoJSON(d.geojson, lat, lon);
      }
    } else {
      dom.layerToggleRainfall.disabled = true;
      dom.layerToggleRainfall.checked = false;
      mapState.activeLayers.rainfall = false;
      dom.layerItemRainfall.className = "layer-toggle-item disabled-layer";
      dom.layerBadgeRainfall.className = "layer-badge-disabled";
      dom.layerBadgeRainfall.textContent = "UNAVAILABLE";

      if (mapState.rainfallLayer) {
        mapState.rainfallLayer.clearLayers();
      }
    }
    updateLegend();
  } catch (err) {
    if (reqId !== currentRequestId) return;
    dom.wlSpatialStatus.textContent = "UNAVAILABLE";
    dom.wlSpatialStatus.className = "status-pill status-high";
    dom.wlSpatialTime.textContent = `Unavailable (${err.message})`;
    dom.layerToggleRainfall.disabled = true;
    dom.layerToggleRainfall.checked = false;
    mapState.activeLayers.rainfall = false;
    dom.layerItemRainfall.className = "layer-toggle-item disabled-layer";
    dom.layerBadgeRainfall.className = "layer-badge-disabled";
    dom.layerBadgeRainfall.textContent = "UNAVAILABLE";
    if (mapState.rainfallLayer) {
      mapState.rainfallLayer.clearLayers();
    }
    updateLegend();
  }
}

// Render Spatial Rainfall Subsection in Waterlogging Panel
function renderSpatialSubsection(d) {
  if (!d) return;

  const st = d.status || "unavailable";
  if (st === "available") {
    dom.wlSpatialStatus.textContent = d.cache_status === "hit" ? "CACHED" : "AVAILABLE";
    dom.wlSpatialStatus.className = "status-pill status-safe";
  } else if (st === "insufficient_coverage") {
    dom.wlSpatialStatus.textContent = "INSUFFICIENT COVERAGE";
    dom.wlSpatialStatus.className = "status-pill status-warning";
  } else {
    dom.wlSpatialStatus.textContent = st.toUpperCase().replace(/_/g, " ");
    dom.wlSpatialStatus.className = "status-pill status-high";
  }

  const win = d.windows || {};
  dom.wlSpatialTime.textContent = d.analysis_time || d.analysis_time_utc || "--";

  if (d.grid) {
    dom.wlSpatialGridInfo.textContent = `Spatial Rainfall Sampling Grid \u2014 0.1\u00b0 request spacing (${d.grid.valid_cells_count || 0}/${d.grid.cell_count || 121} valid)`;
  }

  dom.wlSpatial1h.textContent = formatVal(win["1h"] && win["1h"].value_mm, "mm");
  dom.wlSpatial3h.textContent = formatVal(win["3h"] && win["3h"].value_mm, "mm");
  dom.wlSpatial6h.textContent = formatVal(win["6h"] && win["6h"].value_mm, "mm");
  dom.wlSpatial12h.textContent = formatVal(win["12h"] && win["12h"].value_mm, "mm");
  dom.wlSpatial24h.textContent = formatVal(win["24h"] && win["24h"].value_mm, "mm");
  dom.wlSpatial3d.textContent = formatVal(win["3d"] && win["3d"].value_mm, "mm");
  dom.wlSpatial7d.textContent = formatVal(win["7d"] && win["7d"].value_mm, "mm");

  dom.wlSpatialSourceNote.textContent = "Source: Open-Meteo Recent Hourly Model Data (0.1\u00b0 requested sampling grid). Underlying NWP model selected via Open-Meteo Best Match.";
}

// Render Open-Meteo Spatial Rainfall Provider Card in Bottom Workflow
function renderSpatialWorkflowCard(d) {
  if (!dom.sourceCardSpatialRainfall || !dom.sourceBadgeSpatialRainfall) return;
  if (!d) return;

  const st = d.status || "unavailable";
  if (st === "available") {
    dom.sourceCardSpatialRainfall.className = "source-card status-available";
    dom.sourceBadgeSpatialRainfall.className = "status-badge badge-avail";
    dom.sourceBadgeSpatialRainfall.textContent = d.cache_status === "hit" ? "CACHED" : "AVAILABLE";
    dom.sourceDescSpatialRainfall.textContent = `Open-Meteo recent hourly model data on 0.1\u00b0 requested sampling grid (${d.grid ? d.grid.valid_cells_count : 121}/121 points).`;
  } else {
    dom.sourceCardSpatialRainfall.className = "source-card status-unconfigured";
    dom.sourceBadgeSpatialRainfall.className = "status-badge badge-notconfig";
    dom.sourceBadgeSpatialRainfall.textContent = st.toUpperCase().replace(/_/g, " ");
    dom.sourceDescSpatialRainfall.textContent = "Open-Meteo spatial rainfall data unavailable.";
  }
}

// Asynchronous Non-Blocking Sentinel-1 SAR Fetcher (Phase F)
async function fetchSarWaterContext(lat, lon, reqId) {
  if (!dom.wlSarSection) return;

  dom.wlSarStatus.textContent = "QUERYING...";
  dom.wlSarStatus.className = "status-pill status-neutral";
  dom.wlSarAcquisitionTime.textContent = "Querying Planetary Computer STAC...";
  dom.wlSarSceneAge.textContent = "--";
  dom.wlSarOrbitInfo.textContent = "--";
  dom.wlSarBaselineInfo.textContent = "--";
  dom.wlSarCandidateArea.textContent = "--";
  dom.wlSarPersistentArea.textContent = "--";
  dom.wlSarCoverage.textContent = "--";
  dom.wlSarFraction.textContent = "--";

  try {
    const url = `/api/waterlogging/sar/sentinel1?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_km=10`;
    const res = await fetch(url);
    if (reqId !== currentRequestId) return; // Stale request protection

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const d = json.data || json;
    mapState.latestData.satellite = d;

    renderSarSubsection(d);
    renderSarWorkflowCard(d);

    if (d.status === "available" && d.geojson) {
      dom.layerToggleSatellite.disabled = false;
      dom.layerItemSatellite.className = "layer-toggle-item active-layer";
      dom.layerBadgeSatellite.className = "layer-badge-active";
      dom.layerBadgeSatellite.textContent = "AVAILABLE";

      if (mapState.activeLayers.satellite) {
        renderSarWaterGeoJSON(d.geojson, lat, lon);
      }
    } else {
      dom.layerToggleSatellite.disabled = true;
      dom.layerToggleSatellite.checked = false;
      mapState.activeLayers.satellite = false;
      dom.layerItemSatellite.className = "layer-toggle-item disabled-layer";
      dom.layerBadgeSatellite.className = "layer-badge-disabled";
      dom.layerBadgeSatellite.textContent = (d.status || "UNAVAILABLE").toUpperCase().replace(/_/g, " ");

      if (mapState.satelliteLayer) {
        mapState.satelliteLayer.clearLayers();
      }
    }
    updateLegend();
  } catch (err) {
    if (reqId !== currentRequestId) return;
    dom.wlSarStatus.textContent = "UNAVAILABLE";
    dom.wlSarStatus.className = "status-pill status-high";
    dom.wlSarAcquisitionTime.textContent = `Unavailable (${err.message})`;
    dom.layerToggleSatellite.disabled = true;
    dom.layerToggleSatellite.checked = false;
    mapState.activeLayers.satellite = false;
    dom.layerItemSatellite.className = "layer-toggle-item disabled-layer";
    dom.layerBadgeSatellite.className = "layer-badge-disabled";
    dom.layerBadgeSatellite.textContent = "UNAVAILABLE";
    if (mapState.satelliteLayer) {
      mapState.satelliteLayer.clearLayers();
    }
    updateLegend();
  }
}

// Render Sentinel-1 SAR Subsection in Waterlogging Panel
function renderSarSubsection(d) {
  if (!d) return;

  const st = d.status || "unavailable";
  if (st === "available") {
    dom.wlSarStatus.textContent = d.cache_status === "hit" ? "CACHED" : "AVAILABLE";
    dom.wlSarStatus.className = "status-pill status-safe";
  } else if (st === "insufficient_reference_data") {
    dom.wlSarStatus.textContent = "INSUFFICIENT REFERENCES (<3)";
    dom.wlSarStatus.className = "status-pill status-warning";
  } else if (st === "insufficient_raster_coverage") {
    dom.wlSarStatus.textContent = "INSUFFICIENT COVERAGE";
    dom.wlSarStatus.className = "status-pill status-warning";
  } else {
    dom.wlSarStatus.textContent = st.toUpperCase().replace(/_/g, " ");
    dom.wlSarStatus.className = "status-pill status-high";
  }

  const rec = d.recent_scene || {};
  dom.wlSarAcquisitionTime.textContent = rec.acquisition_datetime ? `${rec.acquisition_datetime} UTC` : "--";

  if (rec.scene_age_days !== undefined) {
    dom.wlSarSceneAge.textContent = `${rec.scene_age_days.toFixed(1)} days (${rec.scene_age_hours ? rec.scene_age_hours.toFixed(0) : '--'} hours)`;
  } else {
    dom.wlSarSceneAge.textContent = "--";
  }

  if (rec.orbit_direction) {
    dom.wlSarOrbitInfo.textContent = `${rec.orbit_direction.toUpperCase()} (Rel. Orbit ${rec.relative_orbit || '--'}, ${rec.platform || 'Sentinel-1'})`;
  } else {
    dom.wlSarOrbitInfo.textContent = "--";
  }

  const baseScenes = d.baseline_scenes || [];
  dom.wlSarBaselineInfo.textContent = `${baseScenes.length} scenes (median composite, dB domain)`;

  const wc = d.water_change || {};
  dom.wlSarCandidateArea.textContent = wc.possible_new_surface_water_area_km2 !== undefined
    ? `${wc.possible_new_surface_water_area_km2.toFixed(3)} km² (${wc.candidate_pixel_count || 0} px)`
    : "--";

  dom.wlSarPersistentArea.textContent = wc.persistent_low_area_km2 !== undefined
    ? `${wc.persistent_low_area_km2.toFixed(3)} km² (${wc.persistent_low_pixel_count || 0} px)`
    : "--";

  const cov = d.coverage || {};
  dom.wlSarCoverage.textContent = cov.coverage_fraction !== undefined
    ? `${(cov.coverage_fraction * 100).toFixed(1)}% (${cov.valid_comparison_pixels || 0} / ${cov.total_roi_pixels || 0} px)`
    : "--";

  dom.wlSarFraction.textContent = wc.candidate_fraction_of_valid_roi !== undefined
    ? `${(wc.candidate_fraction_of_valid_roi * 100).toFixed(3)}% of valid ROI`
    : "--";

  // Populate SAR VV Backscatter row in Section 7
  if (dom.wlSarVvVal) {
    const s1Prov = mapState.latestData.satellite_s1 && mapState.latestData.satellite_s1.provenance
      ? mapState.latestData.satellite_s1.provenance.sentinel1_vv
      : (d.provenance ? d.provenance.sentinel1_vv : null);
    const s1Vv = s1Prov && s1Prov.value !== undefined
      ? s1Prov.value
      : (d.vv_db !== undefined ? d.vv_db : (d.thresholds ? (d.thresholds.vv_low_threshold_db || d.thresholds.otsu_vv_db) : null));

    if (s1Vv !== null && s1Vv !== undefined) {
      dom.wlSarVvVal.textContent = `${Number(s1Vv).toFixed(2)} dB`;
      const isLive = s1Prov ? (s1Prov.quality === "LIVE") : (st === "available");
      if (dom.wlSarVvBadge) {
        dom.wlSarVvBadge.textContent = isLive ? "LIVE" : "FALLBACK";
        dom.wlSarVvBadge.className = `badge-pill ${isLive ? "badge-safe" : "badge-neutral"}`;
      }
      if (dom.wlSarVvMode) {
        const mode = s1Prov && s1Prov.sampling_mode ? s1Prov.sampling_mode : (d.sampling_mode || "RTC");
        dom.wlSarVvMode.textContent = `(${mode})`;
      }
    } else {
      dom.wlSarVvVal.textContent = "-10.5 dB";
      if (dom.wlSarVvBadge) {
        dom.wlSarVvBadge.textContent = "FALLBACK";
        dom.wlSarVvBadge.className = "badge-pill badge-neutral";
      }
      if (dom.wlSarVvMode) {
        dom.wlSarVvMode.textContent = "(baseline)";
      }
    }
  }

  dom.wlSarSourceNote.textContent = d.limitations && d.limitations.length > 0
    ? `Source: Sentinel-1 RTC via Microsoft Planetary Computer. ${d.limitations[0]}`
    : "Source: Sentinel-1 RTC via Microsoft Planetary Computer. Contextual SAR evidence targeting smooth open water.";
}

// Render Sentinel-1 SAR Provider Card in Bottom Workflow
function renderSarWorkflowCard(d) {
  if (!dom.sourceCardSentinel1Sar || !dom.sourceBadgeSentinel1Sar) return;
  if (!d) return;

  const st = d.status || "unavailable";
  if (st === "available") {
    dom.sourceCardSentinel1Sar.className = "source-card status-available";
    dom.sourceBadgeSentinel1Sar.className = "status-badge badge-avail";
    dom.sourceBadgeSentinel1Sar.textContent = d.cache_status === "hit" ? "CACHED" : "AVAILABLE";
    const rec = d.recent_scene || {};
    const wc = d.water_change || {};
    dom.sourceDescSentinel1Sar.textContent = `Sentinel-1 RTC (${rec.platform || 'S1'}, Orbit ${rec.relative_orbit || '--'}). Candidate Water: ${wc.possible_new_surface_water_area_km2 || 0} km².`;
  } else {
    dom.sourceCardSentinel1Sar.className = "source-card status-unconfigured";
    dom.sourceBadgeSentinel1Sar.className = "status-badge badge-notconfig";
    dom.sourceBadgeSentinel1Sar.textContent = st.toUpperCase().replace(/_/g, " ");
    dom.sourceDescSentinel1Sar.textContent = "Sentinel-1 SAR surface water detection unavailable.";
  }
}

// Telemetry Panels Renderer

function renderTelemetry(data) {
  const providerTimes = data.provider_execution_times || {};
  const providerStatus = data.provider_cache_status || {};

  dom.summaryProviderTime.textContent = `Total Fetch: ${formatVal(providerTimes.total_provider_fetch_ms, "ms")}`;
  dom.summaryTotalExecutionTime.textContent = `Total: ${formatVal(data.total_execution_time_ms, "ms")}`;

  dom.providerTelemetryBody.innerHTML = "";

  const providerDisplayNames = {
    "open_meteo_forecast": "Open-Meteo Weather Forecast (Hourly 2d)",
    "open_meteo_air_quality": "Open-Meteo Air Quality (US AQI, PM2.5)",
    "open_meteo_terrain": "Open-Meteo Terrain (13-Point DEM)",
    "open_meteo_flood": "Open-Meteo Flood (River Discharge)",
    "usgs_earthquake": "USGS Earthquake Hazards (500km / 30d)",
    "open_meteo_agriculture": "Open-Meteo Agriculture (Daily 14d)",
    "open_meteo_waterlogging": "Open-Meteo Waterlogging (Hourly 7d)",
    "gdacs": "GDACS Active Hazard Alerts (Cyclone/Volcano)"
  };

  Object.entries(providerDisplayNames).forEach(([key, label]) => {
    const stat = providerStatus[key] || {};
    const ms = providerTimes[`${key}_ms`];
    const isCached = stat.cached === true;
    const httpCount = stat.http_requests !== undefined ? stat.http_requests : (isCached ? 0 : 1);

    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.textContent = label;
    tr.appendChild(tdName);

    const tdMs = document.createElement("td");
    tdMs.textContent = formatVal(ms !== undefined ? ms : stat.fetch_ms, "ms");
    tr.appendChild(tdMs);

    const tdCached = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `status-pill ${isCached ? 'status-success' : 'status-neutral'}`;
    badge.textContent = isCached ? "Cache Hit" : "Cache Miss";
    tdCached.appendChild(badge);
    tr.appendChild(tdCached);

    const tdHttp = document.createElement("td");
    tdHttp.textContent = `${httpCount} HTTP`;
    tr.appendChild(tdHttp);

    dom.providerTelemetryBody.appendChild(tr);
  });

  const execTimes = data.execution_times || {};
  dom.modulePerformanceBody.innerHTML = "";

  const moduleNames = [
    { key: "weather", label: "Weather Intelligence (WeatherGPT)" },
    { key: "disaster", label: "Disaster Management (XGBoost)" },
    { key: "agriculture", label: "Agriculture Advisory (LightGBM)" },
    { key: "waterlogging", label: "Waterlogging Hazard (Random Forest)" },
  ];

  moduleNames.forEach(({ key, label }) => {
    const mod = execTimes[key] || {};
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.textContent = label;
    tr.appendChild(tdName);

    const tdNet = document.createElement("td");
    tdNet.textContent = formatVal(mod.network_ms, "ms");
    tr.appendChild(tdNet);

    const tdFb = document.createElement("td");
    tdFb.textContent = formatVal(mod.feature_build_ms, "ms");
    tr.appendChild(tdFb);

    const tdInf = document.createElement("td");
    tdInf.textContent = formatVal(mod.inference_ms, "ms");
    tr.appendChild(tdInf);

    const tdTot = document.createElement("td");
    tdTot.style.fontWeight = "700";
    tdTot.textContent = formatVal(mod.total_ms || execTimes[`${key}_ms`], "ms");
    tr.appendChild(tdTot);

    dom.modulePerformanceBody.appendChild(tr);
  });
}

// =============================================================================
// 6. LOCATION SEARCH & GEOLOCATION WORKFLOW
// =============================================================================

dom.placeSearchInput.addEventListener("input", () => {
  dom.placeSearchResults.style.display = "none";
  dom.validationMsg.style.display = "none";
});

async function performLocationSearch() {
  const q = dom.placeSearchInput.value.trim();
  if (!q) {
    dom.validationMsg.textContent = "Please enter a village, town, city, district, or 6-digit PIN code in India.";
    dom.validationMsg.style.display = "block";
    dom.placeSearchResults.style.display = "none";
    return;
  }

  dom.validationMsg.style.display = "none";
  dom.searchSpinner.style.display = "inline-block";
  dom.searchLocationBtn.disabled = true;

  try {
    const res = await fetch(`/api/sources/geocode?q=${encodeURIComponent(q)}&limit=5`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    dom.placeSearchResults.innerHTML = "";

    if (data && data.success && Array.isArray(data.data) && data.data.length > 0) {
      data.data.forEach(item => {
        const div = document.createElement("div");
        div.className = "search-result-item";

        const title = document.createElement("div");
        title.className = "search-result-title";
        title.textContent = item.name || item.display_name.split(",")[0];

        const sub = document.createElement("div");
        sub.className = "search-result-subtitle";
        sub.textContent = item.display_name;

        div.appendChild(title);
        div.appendChild(sub);

        div.addEventListener("click", () => {
          const exactLat = Number(item.lat);
          const exactLon = Number(item.lon);
          selectedLocation = {
            lat: exactLat,
            lon: exactLon,
            name: item.display_name,
            isAreaLevel: (item.type === "administrative" || item.class === "boundary")
          };

          dom.latInput.value = exactLat;
          dom.lonInput.value = exactLon;
          dom.placeSearchInput.value = item.display_name;
          dom.selectedPlaceName.textContent = item.display_name;
          dom.selectedLocationCard.style.display = "inline-flex";
          dom.placeSearchResults.style.display = "none";
          dom.validationMsg.style.display = "none";

          // Show disclaimer tag clearly
          dom.pointAnalysisDisclaimer.textContent = "Point analysis at selected/resolved coordinate";

          // Auto fly map to selected point
          if (mapState.map) {
            mapState.map.flyTo([exactLat, exactLon], 12, { duration: 1.0 });
          }

          // Trigger unified analysis for newly selected location
          runAnalysis();
        });
        dom.placeSearchResults.appendChild(div);
      });
      dom.placeSearchResults.style.display = "block";
    } else {
      dom.placeSearchResults.style.display = "none";
      dom.validationMsg.textContent = `No locations found for "${q}". Please search for an Indian village, town, city, district, or PIN code.`;
      dom.validationMsg.style.display = "block";
    }
  } catch (err) {
    dom.placeSearchResults.style.display = "none";
    dom.validationMsg.textContent = "Location search failed. Please check network connection.";
    dom.validationMsg.style.display = "block";
  } finally {
    dom.searchSpinner.style.display = "none";
    dom.searchLocationBtn.disabled = false;
  }
}

dom.searchLocationBtn.addEventListener("click", performLocationSearch);
dom.placeSearchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    performLocationSearch();
  }
});

// GPS Current Location Handling
dom.useLocationBtn.addEventListener("click", () => {
  if (!navigator.geolocation) {
    dom.validationMsg.textContent = "Geolocation is not supported by your browser. Please search by name or PIN code.";
    dom.validationMsg.style.display = "block";
    return;
  }

  dom.gpsBtnText.textContent = "Locating...";
  dom.useLocationBtn.disabled = true;
  dom.validationMsg.style.display = "none";

  navigator.geolocation.getCurrentPosition(
    async (pos) => {
      const exactLat = Number(pos.coords.latitude);
      const exactLon = Number(pos.coords.longitude);

      selectedLocation = {
        lat: exactLat,
        lon: exactLon,
        name: `Current Location (${exactLat.toFixed(4)}, ${exactLon.toFixed(4)})`,
        isAreaLevel: false
      };

      dom.latInput.value = exactLat;
      dom.lonInput.value = exactLon;
      dom.placeSearchInput.value = `Current Location (${exactLat.toFixed(4)}, ${exactLon.toFixed(4)})`;
      dom.selectedPlaceName.textContent = selectedLocation.name;
      dom.selectedLocationCard.style.display = "inline-flex";
      dom.placeSearchResults.style.display = "none";
      dom.gpsBtnText.textContent = "GPS Location";
      dom.useLocationBtn.disabled = false;

      // Reverse geocode to resolve place name
      try {
        const revRes = await fetch(`/api/sources/reverse-geocode?lat=${exactLat}&lon=${exactLon}`);
        if (revRes.ok) {
          const revData = await revRes.json();
          if (revData && revData.success && revData.data && revData.data.display_name) {
            selectedLocation.name = revData.data.display_name;
            dom.selectedPlaceName.textContent = revData.data.display_name;
            dom.placeSearchInput.value = revData.data.display_name;
          }
        }
      } catch {
        // Reverse geocoding failure must not invalidate coordinates
      }

      if (mapState.map) {
        mapState.map.flyTo([exactLat, exactLon], 12, { duration: 1.0 });
      }

      // Trigger unified analysis for GPS location
      runAnalysis();
    },
    (err) => {
      dom.gpsBtnText.textContent = "GPS Location";
      dom.useLocationBtn.disabled = false;
      let msg = "Could not retrieve current location. Please search by village, town, city, or PIN code.";
      if (err.code === 1) msg = "Location permission was denied. Please search manually.";
      else if (err.code === 2) msg = "Location position unavailable. Please search manually.";
      else if (err.code === 3) msg = "Location request timed out. Please search manually.";
      dom.validationMsg.textContent = msg;
      dom.validationMsg.style.display = "block";
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
  );
});

// =============================================================================
// 6B. PAN-INDIA HIERARCHICAL SPATIAL RISK NAVIGATION & GRID RENDERING
// =============================================================================

let statesCatalogCache = [];

async function loadStatesCatalog() {
  try {
    const res = await fetch("/api/risk/states");
    if (!res.ok) return;
    const json = await res.json();
    if (json.success && Array.isArray(json.data)) {
      statesCatalogCache = json.data;
      if (dom.navStateSelect) {
        dom.navStateSelect.innerHTML = '<option value="">-- Select State / UT --</option>';
        statesCatalogCache.forEach(st => {
          const opt = document.createElement("option");
          opt.value = st.name;
          opt.textContent = `${st.name} (${st.type === "UT" ? "UT" : "State"})`;
          dom.navStateSelect.appendChild(opt);
        });
      }
    }
  } catch (err) {
    console.warn("Could not load states catalog:", err);
  }
}

function updateActiveViewBadge(tier, title, subtitle) {
  if (dom.activeViewBadge) {
    dom.activeViewBadge.textContent = tier.toUpperCase();
  }
  if (dom.spatialRegionTitle) {
    dom.spatialRegionTitle.textContent = title;
  }
  if (dom.spatialSubTitle) {
    dom.spatialSubTitle.textContent = subtitle || "";
  }
  [dom.viewBtnIndia, dom.viewBtnState, dom.viewBtnDistrict, dom.viewBtnGrid].forEach(btn => {
    if (btn) btn.classList.remove("active");
  });
  if (tier === "INDIA" && dom.viewBtnIndia) dom.viewBtnIndia.classList.add("active");
  if (tier === "STATE" && dom.viewBtnState) dom.viewBtnState.classList.add("active");
  if (tier === "DISTRICT" && dom.viewBtnDistrict) dom.viewBtnDistrict.classList.add("active");
  if (tier === "GRID" && dom.viewBtnGrid) dom.viewBtnGrid.classList.add("active");
}

async function loadIndiaView() {
  mapState.currentView = "INDIA";
  updateActiveViewBadge("INDIA", "PAN-INDIA RISK OVERVIEW", "National state-level synthesized multi-hazard context (28 States + 8 UTs)");

  // Clear local grid and district layers
  if (mapState.wlGridLayer) mapState.wlGridLayer.clearLayers();
  if (mapState.drainGridLayer) mapState.drainGridLayer.clearLayers();
  if (mapState.hotspotGridLayer) mapState.hotspotGridLayer.clearLayers();
  if (mapState.districtLayer) mapState.districtLayer.clearLayers();

  if (dom.spatialCellCountBadge) {
    dom.spatialCellCountBadge.textContent = "Loading National States...";
  }

  // Zoom to national India extent
  if (mapState.map) {
    mapState.map.flyTo([22.5, 82.0], 5, { duration: 1.0 });
  }

  try {
    const res = await fetch("/api/risk/india");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (!json.success || !json.data) return;

    mapState.indiaGeoJSONData = json.data;
    renderIndiaStateMarkers(json.data);
    updateIndiaSummaryPanel(json.data);
  } catch (err) {
    console.error("Failed to load India view:", err);
    if (dom.spatialCellCountBadge) dom.spatialCellCountBadge.textContent = "National Data Unavailable";
  }
}

function renderIndiaStateMarkers(data) {
  if (!mapState.stateLayer) return;
  mapState.stateLayer.clearLayers();

  const features = (data.geojson && data.geojson.features) || [];
  features.forEach(f => {
    const coords = f.geometry && f.geometry.coordinates;
    if (!coords || coords.length < 2) return;
    const [lon, lat] = coords;
    const p = f.properties || {};
    const risk = p.overall_risk || "LOW";
    const color = getRiskHexColor(risk);

    const marker = L.circleMarker([lat, lon], {
      radius: 11,
      fillColor: color,
      color: "#ffffff",
      weight: 1.5,
      opacity: 0.95,
      fillOpacity: 0.8
    });

    const popupHtml = `
    <div style="min-width: 200px; font-family: inherit;">
      <h4 style="margin: 0 0 4px 0; color: #38bdf8; font-size: 14px;">${escapeHtml(p.state)}</h4>
      <div style="font-size: 11px; margin-bottom: 6px;">
        <span class="badge-pill ${getRiskBadgeClass(risk)}">${risk} RISK</span> • ${escapeHtml(p.type || 'State')}
      </div>
      <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 2px;"><strong>Capital:</strong> ${escapeHtml(p.capital || '--')}</div>
      <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 2px;"><strong>Disaster Alerts:</strong> ${escapeHtml(p.disaster_context || 'None active')}</div>
      <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 8px;"><strong>Rainfall:</strong> ${escapeHtml(p.rainfall_context || 'Nominal')}</div>
      <button class="btn btn-sm btn-primary w-100" style="padding: 4px 8px; font-size: 11px;" onclick="loadStateView('${escapeHtml(p.state)}')">Drill Down to State &rarr;</button>
    </div>
  `;

    marker.bindPopup(popupHtml);
    marker.bindTooltip(`<strong>${escapeHtml(p.state)}</strong>: ${risk}`, { direction: "top", offset: [0, -8] });
    marker.addTo(mapState.stateLayer);
  });
}

function updateIndiaSummaryPanel(data) {
  if (!dom.spatialSummaryCard) return;
  dom.spatialSummaryCard.style.display = "block";
  if (dom.spatialCellCountBadge) dom.spatialCellCountBadge.textContent = `${data.total_states || 36} States & UTs`;
  if (dom.spatialDataQualityBadge) {
    dom.spatialDataQualityBadge.textContent = "SYNTHESIZED EVIDENCE";
    dom.spatialDataQualityBadge.className = "status-pill status-safe";
  }
  const summary = data.summary || {};
  const total = summary.total || 36;
  const crit = summary.critical || 0;
  const high = summary.high || 0;
  const mod = summary.moderate || 0;
  const low = summary.low || 0;

  if (dom.wlDistLow) dom.wlDistLow.textContent = `${((low / total) * 100).toFixed(0)}%`;
  if (dom.wlDistMod) dom.wlDistMod.textContent = `${((mod / total) * 100).toFixed(0)}%`;
  if (dom.wlDistHigh) dom.wlDistHigh.textContent = `${((high / total) * 100).toFixed(0)}%`;
  if (dom.wlDistCrit) dom.wlDistCrit.textContent = `${((crit / total) * 100).toFixed(0)}%`;

  if (dom.wlBarLow) dom.wlBarLow.style.width = `${(low / total) * 100}%`;
  if (dom.wlBarMod) dom.wlBarMod.style.width = `${(mod / total) * 100}%`;
  if (dom.wlBarHigh) dom.wlBarHigh.style.width = `${(high / total) * 100}%`;
  if (dom.wlBarCrit) dom.wlBarCrit.style.width = `${(crit / total) * 100}%`;

  if (dom.drDistLow) dom.drDistLow.textContent = `${((low / total) * 100).toFixed(0)}%`;
  if (dom.drDistMod) dom.drDistMod.textContent = `${((mod / total) * 100).toFixed(0)}%`;
  if (dom.drDistHigh) dom.drDistHigh.textContent = `${((high / total) * 100).toFixed(0)}%`;
  if (dom.drDistCrit) dom.drDistCrit.textContent = `${((crit / total) * 100).toFixed(0)}%`;

  if (dom.drBarLow) dom.drBarLow.style.width = `${(low / total) * 100}%`;
  if (dom.drBarMod) dom.drBarMod.style.width = `${(mod / total) * 100}%`;
  if (dom.drBarHigh) dom.drBarHigh.style.width = `${(high / total) * 100}%`;
  if (dom.drBarCrit) dom.drBarCrit.style.width = `${(crit / total) * 100}%`;

  if (dom.hotspotCountVal) dom.hotspotCountVal.textContent = crit + high;
  if (dom.hotspotSubNote) dom.hotspotSubNote.textContent = `${crit} Critical • ${high} High States`;
  if (dom.potentialCountVal) dom.potentialCountVal.textContent = mod;

  if (dom.metaResolution) dom.metaResolution.textContent = "State-Level Granularity";
  if (dom.metaInferenceTime) dom.metaInferenceTime.textContent = "National Real-Time Synthesis";
  if (dom.metaTotalLatency) dom.metaTotalLatency.textContent = "< 1.5s National Latency";

  if (dom.hotspotsSection) dom.hotspotsSection.style.display = "none";
}

async function loadStateView(stateName) {
  if (!stateName) return;
  mapState.currentView = "STATE";
  updateActiveViewBadge("STATE", `${stateName.toUpperCase()} RISK OVERVIEW`, "District-level summarized risk and hazard context");

  if (dom.navStateSelect) dom.navStateSelect.value = stateName;

  // Clear previous layers
  if (mapState.stateLayer) mapState.stateLayer.clearLayers();
  if (mapState.districtLayer) mapState.districtLayer.clearLayers();
  if (mapState.wlGridLayer) mapState.wlGridLayer.clearLayers();
  if (mapState.drainGridLayer) mapState.drainGridLayer.clearLayers();
  if (mapState.hotspotGridLayer) mapState.hotspotGridLayer.clearLayers();

  if (dom.spatialCellCountBadge) {
    dom.spatialCellCountBadge.textContent = "Loading Districts...";
  }

  try {
    const res = await fetch(`/api/risk/state?state=${encodeURIComponent(stateName)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (!json.success || !json.data) return;

    const data = json.data;
    mapState.stateGeoJSONData = data;

    // Zoom to state bbox if available
    if (data.bbox && mapState.map) {
      mapState.map.fitBounds([
        [data.bbox.min_lat, data.bbox.min_lon],
        [data.bbox.max_lat, data.bbox.max_lon]
      ], { padding: [25, 25], maxZoom: 9 });
    }

    // Populate district dropdown
    if (dom.navDistrictSelect && Array.isArray(data.districts)) {
      dom.navDistrictSelect.innerHTML = '<option value="">-- Select District --</option>';
      data.districts.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.district;
        opt.textContent = `${d.district} (${d.overall_risk})`;
        dom.navDistrictSelect.appendChild(opt);
      });
    }

    renderDistrictMarkers(data);
    updateStateSummaryPanel(data);
  } catch (err) {
    console.error("Failed to load State view:", err);
    if (dom.spatialCellCountBadge) dom.spatialCellCountBadge.textContent = "State Data Unavailable";
  }
}

function renderDistrictMarkers(data) {
  if (!mapState.districtLayer) return;
  mapState.districtLayer.clearLayers();

  const districts = data.districts || [];
  districts.forEach(d => {
    const risk = d.overall_risk || "LOW";
    const color = getRiskHexColor(risk);

    const marker = L.circleMarker([d.lat, d.lon], {
      radius: 9,
      fillColor: color,
      color: "#ffffff",
      weight: 1.5,
      opacity: 0.9,
      fillOpacity: 0.8
    });

    const popupHtml = `
    <div style="min-width: 200px; font-family: inherit;">
      <h4 style="margin: 0 0 4px 0; color: #38bdf8; font-size: 14px;">${escapeHtml(d.district)}</h4>
      <div style="font-size: 11px; margin-bottom: 6px;">
        <span class="badge-pill ${getRiskBadgeClass(risk)}">${risk}</span> • ${escapeHtml(d.state)}
      </div>
      <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 2px;"><strong>Waterlogging:</strong> ${escapeHtml(d.waterlogging_summary || 'Nominal')}</div>
      <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 2px;"><strong>Drainage:</strong> ${escapeHtml(d.drainage_summary || 'Adequate')}</div>
      <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 8px;"><strong>Disaster Alerts:</strong> ${escapeHtml(d.disaster_context || 'None')}</div>
      <button class="btn btn-sm btn-primary w-100" style="padding: 4px 8px; font-size: 11px;" onclick="loadDistrictView('${escapeHtml(d.state)}', '${escapeHtml(d.district)}')">Analyze District Grid &rarr;</button>
    </div>
  `;

    marker.bindPopup(popupHtml);
    marker.bindTooltip(`<strong>${escapeHtml(d.district)}</strong>: ${risk}`, { direction: "top", offset: [0, -8] });
    marker.addTo(mapState.districtLayer);
  });
}

function updateStateSummaryPanel(data) {
  if (!dom.spatialSummaryCard) return;
  dom.spatialSummaryCard.style.display = "block";
  const districts = data.districts || [];
  const total = districts.length;
  if (dom.spatialCellCountBadge) dom.spatialCellCountBadge.textContent = `${total} Districts`;
  if (dom.spatialDataQualityBadge) {
    dom.spatialDataQualityBadge.textContent = "DISTRICT HARVEST";
    dom.spatialDataQualityBadge.className = "status-pill status-safe";
  }

  let low = 0, mod = 0, high = 0, crit = 0;
  districts.forEach(d => {
    if (d.overall_risk === "CRITICAL") crit++;
    else if (d.overall_risk === "HIGH") high++;
    else if (d.overall_risk === "MODERATE") mod++;
    else low++;
  });

  const pct = (c) => total > 0 ? (c / total) * 100 : 0;

  if (dom.wlDistLow) dom.wlDistLow.textContent = `${pct(low).toFixed(0)}%`;
  if (dom.wlDistMod) dom.wlDistMod.textContent = `${pct(mod).toFixed(0)}%`;
  if (dom.wlDistHigh) dom.wlDistHigh.textContent = `${pct(high).toFixed(0)}%`;
  if (dom.wlDistCrit) dom.wlDistCrit.textContent = `${pct(crit).toFixed(0)}%`;

  if (dom.wlBarLow) dom.wlBarLow.style.width = `${pct(low)}%`;
  if (dom.wlBarMod) dom.wlBarMod.style.width = `${pct(mod)}%`;
  if (dom.wlBarHigh) dom.wlBarHigh.style.width = `${pct(high)}%`;
  if (dom.wlBarCrit) dom.wlBarCrit.style.width = `${pct(crit)}%`;

  if (dom.drDistLow) dom.drDistLow.textContent = `${pct(low).toFixed(0)}%`;
  if (dom.drDistMod) dom.drDistMod.textContent = `${pct(mod).toFixed(0)}%`;
  if (dom.drDistHigh) dom.drDistHigh.textContent = `${pct(high).toFixed(0)}%`;
  if (dom.drDistCrit) dom.drDistCrit.textContent = `${pct(crit).toFixed(0)}%`;

  if (dom.drBarLow) dom.drBarLow.style.width = `${pct(low)}%`;
  if (dom.drBarMod) dom.drBarMod.style.width = `${pct(mod)}%`;
  if (dom.drBarHigh) dom.drBarHigh.style.width = `${pct(high)}%`;
  if (dom.drBarCrit) dom.drBarCrit.style.width = `${pct(crit)}%`;

  if (dom.hotspotCountVal) dom.hotspotCountVal.textContent = crit + high;
  if (dom.hotspotSubNote) dom.hotspotSubNote.textContent = `${crit} Critical • ${high} High Districts`;
  if (dom.potentialCountVal) dom.potentialCountVal.textContent = mod;

  if (dom.metaResolution) dom.metaResolution.textContent = "District-Level Granularity";
  if (dom.metaInferenceTime) dom.metaInferenceTime.textContent = "District Aggregation";
  if (dom.metaTotalLatency) dom.metaTotalLatency.textContent = "~200 ms";

  if (dom.hotspotsSection) dom.hotspotsSection.style.display = "none";
}

async function loadDistrictView(stateName, districtName) {
  if (!stateName || !districtName) return;
  mapState.currentView = "DISTRICT";
  updateActiveViewBadge("DISTRICT", `${districtName.toUpperCase()}, ${stateName.toUpperCase()}`, "Resolving district boundary and generating high-resolution ML grid...");

  if (dom.navStateSelect) dom.navStateSelect.value = stateName;
  if (dom.navDistrictSelect) dom.navDistrictSelect.value = districtName;

  try {
    const res = await fetch(`/api/risk/district?state=${encodeURIComponent(stateName)}&district=${encodeURIComponent(districtName)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (!json.success || !json.data) return;

    const d = json.data;
    selectedLocation = {
      lat: d.lat,
      lon: d.lon,
      name: `${d.district}, ${d.state}, India`,
      isAreaLevel: true
    };
    if (dom.latInput) dom.latInput.value = d.lat;
    if (dom.lonInput) dom.lonInput.value = d.lon;
    if (dom.selectedPlaceName) dom.selectedPlaceName.textContent = selectedLocation.name;
    if (dom.placeSearchInput) dom.placeSearchInput.value = selectedLocation.name;

    if (d.bbox && mapState.map) {
      mapState.map.fitBounds([
        [d.bbox.min_lat, d.bbox.min_lon],
        [d.bbox.max_lat, d.bbox.max_lon]
      ], { padding: [30, 30], maxZoom: 13 });
    }

    // Generate local risk grid for district
    const resM = Number((dom.navGridResolution && dom.navGridResolution.value) || 500);
    generateLocalRiskGrid(d.lat, d.lon, 5000, resM, stateName, districtName);
  } catch (err) {
    console.error("Failed to load District view:", err);
  }
}

async function generateLocalRiskGrid(lat, lon, radius_m = 5000, grid_size_m = 500, state = null, district = null) {
  mapState.currentView = "GRID";
  updateActiveViewBadge("GRID", "LOCAL MULTI-RISK ML GRID", `High-resolution ${grid_size_m}m inference grid (${(radius_m / 1000).toFixed(0)}km radius ROI)`);

  if (dom.spatialCellCountBadge) {
    dom.spatialCellCountBadge.textContent = "Computing ML Grid...";
  }

  // Clear previous layers
  if (mapState.wlGridLayer) mapState.wlGridLayer.clearLayers();
  if (mapState.drainGridLayer) mapState.drainGridLayer.clearLayers();
  if (mapState.hotspotGridLayer) mapState.hotspotGridLayer.clearLayers();
  if (mapState.stateLayer) mapState.stateLayer.clearLayers();
  if (mapState.districtLayer) mapState.districtLayer.clearLayers();

  try {
    const payload = {
      lat: Number(lat),
      lon: Number(lon),
      radius_m: Number(radius_m),
      grid_size_m: Number(grid_size_m)
    };
    if (state) payload.state = state;
    if (district) payload.district = district;

    const res = await fetch("/api/risk/grid", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const json = await res.json();
    if (!json.success || !json.data) throw new Error("Invalid grid risk response");

    mapState.gridGeoJSONData = json.data;
    renderGridLayers(json.data);
    updateSpatialSummaryPanel(json.data);
    renderHotspotsTable(json.data.summary ? json.data.summary.top_hotspots : []);
  } catch (err) {
    console.error("Failed to generate local risk grid:", err);
    if (dom.spatialCellCountBadge) {
      dom.spatialCellCountBadge.textContent = "Grid Generation Failed";
    }
  }
}

function renderGridLayers(data) {
  if (!mapState.wlGridLayer || !mapState.drainGridLayer || !mapState.hotspotGridLayer) return;

  mapState.wlGridLayer.clearLayers();
  mapState.drainGridLayer.clearLayers();
  mapState.hotspotGridLayer.clearLayers();

  const geojson = data.geojson || (data.features ? data : null);
  if (!geojson || !geojson.features) return;

  // 1. Waterlogging Grid Layer
  const wlLayer = L.geoJSON(geojson, {
    style: (feature) => {
      const p = feature.properties || {};
      const risk = p.waterlogging_risk || "NO DATA";
      const color = risk === "NO DATA" ? "#64748b" : getRiskHexColor(risk);
      return {
        fillColor: color,
        weight: 1,
        opacity: 0.6,
        color: "#1e293b",
        fillOpacity: 0.5
      };
    },
    onEachFeature: (feature, layer) => {
      layer.bindPopup(createGridCellPopup(feature.properties));
    }
  });
  wlLayer.addTo(mapState.wlGridLayer);

  // 2. Drainage Grid Layer
  const drainLayer = L.geoJSON(geojson, {
    style: (feature) => {
      const p = feature.properties || {};
      const risk = p.drainage_risk || "NO DATA";
      const color = risk === "NO DATA" ? "#64748b" : getRiskHexColor(risk);
      return {
        fillColor: color,
        weight: 1,
        opacity: 0.6,
        color: "#1e293b",
        fillOpacity: 0.5
      };
    },
    onEachFeature: (feature, layer) => {
      layer.bindPopup(createGridCellPopup(feature.properties));
    }
  });
  drainLayer.addTo(mapState.drainGridLayer);

  // 3. Hotspot Grid Layer (only critical/high/potential hotspots)
  const hotspotFeatures = geojson.features.filter(f => {
    const st = f.properties && f.properties.combined_hotspot_status;
    return st && st !== "NORMAL";
  });

  if (hotspotFeatures.length > 0) {
    const hotspotGeoJSON = {
      type: "FeatureCollection",
      features: hotspotFeatures
    };

    const hsLayer = L.geoJSON(hotspotGeoJSON, {
      style: (feature) => {
        const p = feature.properties || {};
        const st = p.combined_hotspot_status;
        const color = st === "CRITICAL_HOTSPOT" ? "#ef4444" : st === "HIGH_RISK_HOTSPOT" ? "#f97316" : "#f59e0b";
        return {
          fillColor: color,
          weight: 2,
          opacity: 1.0,
          color: "#ffffff",
          dashArray: "3, 3",
          fillOpacity: 0.65
        };
      },
      onEachFeature: (feature, layer) => {
        layer.bindPopup(createGridCellPopup(feature.properties));
      }
    });
    hsLayer.addTo(mapState.hotspotGridLayer);
  }

  // Ensure active layer state matches checkboxes
  if (dom.layerToggleWlGrid && !dom.layerToggleWlGrid.checked) {
    mapState.map.removeLayer(mapState.wlGridLayer);
  }
  if (dom.layerToggleDrainGrid && !dom.layerToggleDrainGrid.checked) {
    mapState.map.removeLayer(mapState.drainGridLayer);
  }
  if (dom.layerToggleHotspots && !dom.layerToggleHotspots.checked) {
    mapState.map.removeLayer(mapState.hotspotGridLayer);
  }

  // Auto-fit to grid bounds
  if (mapState.map && wlLayer.getBounds().isValid()) {
    mapState.map.fitBounds(wlLayer.getBounds(), { padding: [20, 20], maxZoom: 14 });
  }
}

function createGridCellPopup(props) {
  const wlProb = props.waterlogging_probability != null ? `${(props.waterlogging_probability * 100).toFixed(1)}%` : "--";
  const drProb = props.drainage_probability != null ? `${(props.drainage_probability * 100).toFixed(1)}%` : "--";
  const wlRisk = props.waterlogging_risk || "NO DATA";
  const drRisk = props.drainage_risk || "NO DATA";
  const elev = props.elevation != null ? `${props.elevation.toFixed(1)} m` : "--";
  const slope = props.slope != null ? `${props.slope.toFixed(1)}°` : "--";
  const rain = props.rain_max != null ? `${props.rain_max.toFixed(1)} mm` : "--";
  const s1 = props.sentinel1_vv != null ? `${props.sentinel1_vv.toFixed(1)} dB` : "--";
  const ndvi = props.NDVI != null ? props.NDVI.toFixed(3) : "--";
  const ndbi = props.NDBI != null ? props.NDBI.toFixed(3) : "--";
  const dq = props.data_quality || "FALLBACK_USED";
  const hotspot = props.combined_hotspot_status || "NORMAL";

  const hotspotBadge = hotspot === "CRITICAL_HOTSPOT"
    ? '<span class="badge-pill badge-danger">CRITICAL HOTSPOT</span>'
    : hotspot === "HIGH_RISK_HOTSPOT"
      ? '<span class="badge-pill badge-warning">HIGH RISK HOTSPOT</span>'
      : hotspot === "POTENTIAL_HOTSPOT"
        ? '<span class="badge-pill badge-info">POTENTIAL HOTSPOT</span>'
        : '<span class="badge-pill badge-safe">NOMINAL</span>';

  return `
  <div class="grid-cell-popup">
    <div class="popup-header">
      <strong>Cell #${props.grid_id != null ? props.grid_id : '--'}</strong>
      <span class="badge-pill ${getRiskBadgeClass(wlRisk)}">${wlRisk}</span>
    </div>
    <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px;">
      ${props.center_lat ? props.center_lat.toFixed(4) : '--'}, ${props.center_lon ? props.center_lon.toFixed(4) : '--'}
      ${props.district ? `• ${props.district}` : ''}
    </div>
    <div style="margin-bottom: 6px;">
      <strong>Hotspot Status:</strong> ${hotspotBadge}
    </div>
    <div class="popup-grid-metrics">
      <div><span class="lbl">Waterlogging:</span> <strong>${wlProb}</strong> (${wlRisk})</div>
      <div><span class="lbl">Drainage:</span> <strong>${drProb}</strong> (${drRisk})</div>
      <div><span class="lbl">Rain Max:</span> <strong>${rain}</strong></div>
      <div><span class="lbl">Elevation:</span> <strong>${elev}</strong></div>
      <div><span class="lbl">Slope:</span> <strong>${slope}</strong></div>
      <div><span class="lbl">Sentinel-1 VV:</span> <strong>${s1}</strong></div>
      <div><span class="lbl">NDVI:</span> <strong>${ndvi}</strong></div>
      <div><span class="lbl">NDBI:</span> <strong>${ndbi}</strong></div>
    </div>
    <div style="margin-top: 6px; font-size: 10px; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
      Data Quality: <span class="badge-pill ${dq === 'FULL_DATA' ? 'badge-safe' : 'badge-neutral'}">${dq}</span>
    </div>
  </div>
`;
}

function updateSpatialSummaryPanel(data) {
  if (!dom.spatialSummaryCard) return;
  dom.spatialSummaryCard.style.display = "block";

  const sum = data.summary || {};
  const total = sum.total_cells || 0;
  if (dom.spatialCellCountBadge) {
    dom.spatialCellCountBadge.textContent = `${total} Cells Computed`;
  }
  if (dom.spatialDataQualityBadge) {
    const dq = data.data_quality || "FULL_DATA";
    dom.spatialDataQualityBadge.textContent = dq;
    dom.spatialDataQualityBadge.className = `status-pill ${dq === 'FULL_DATA' ? 'status-safe' : 'status-neutral'}`;
  }

  // Waterlogging distribution
  const wl = sum.waterlogging_distribution || {};
  const wlLow = wl.LOW || { count: 0, percentage: 0 };
  const wlMod = wl.MODERATE || { count: 0, percentage: 0 };
  const wlHigh = wl.HIGH || { count: 0, percentage: 0 };
  const wlCrit = wl.CRITICAL || { count: 0, percentage: 0 };

  if (dom.wlDistLow) dom.wlDistLow.textContent = `${wlLow.percentage.toFixed(0)}%`;
  if (dom.wlDistMod) dom.wlDistMod.textContent = `${wlMod.percentage.toFixed(0)}%`;
  if (dom.wlDistHigh) dom.wlDistHigh.textContent = `${wlHigh.percentage.toFixed(0)}%`;
  if (dom.wlDistCrit) dom.wlDistCrit.textContent = `${wlCrit.percentage.toFixed(0)}%`;

  if (dom.wlBarLow) dom.wlBarLow.style.width = `${wlLow.percentage}%`;
  if (dom.wlBarMod) dom.wlBarMod.style.width = `${wlMod.percentage}%`;
  if (dom.wlBarHigh) dom.wlBarHigh.style.width = `${wlHigh.percentage}%`;
  if (dom.wlBarCrit) dom.wlBarCrit.style.width = `${wlCrit.percentage}%`;

  // Drainage distribution
  const dr = sum.drainage_distribution || {};
  const drLow = dr.LOW || { count: 0, percentage: 0 };
  const drMod = dr.MODERATE || { count: 0, percentage: 0 };
  const drHigh = dr.HIGH || { count: 0, percentage: 0 };
  const drCrit = dr.CRITICAL || { count: 0, percentage: 0 };

  if (dom.drDistLow) dom.drDistLow.textContent = `${drLow.percentage.toFixed(0)}%`;
  if (dom.drDistMod) dom.drDistMod.textContent = `${drMod.percentage.toFixed(0)}%`;
  if (dom.drDistHigh) dom.drDistHigh.textContent = `${drHigh.percentage.toFixed(0)}%`;
  if (dom.drDistCrit) dom.drDistCrit.textContent = `${drCrit.percentage.toFixed(0)}%`;

  if (dom.drBarLow) dom.drBarLow.style.width = `${drLow.percentage}%`;
  if (dom.drBarMod) dom.drBarMod.style.width = `${drMod.percentage}%`;
  if (dom.drBarHigh) dom.drBarHigh.style.width = `${drHigh.percentage}%`;
  if (dom.drBarCrit) dom.drBarCrit.style.width = `${drCrit.percentage}%`;

  // Hotspots count
  const hot = sum.hotspots || {};
  const critHot = hot.critical_hotspots || 0;
  const highHot = hot.high_risk_hotspots || 0;
  const potHot = hot.potential_hotspots || 0;
  const combinedHot = critHot + highHot;

  if (dom.hotspotCountVal) dom.hotspotCountVal.textContent = combinedHot;
  if (dom.hotspotSubNote) dom.hotspotSubNote.textContent = `${critHot} Critical • ${highHot} High-Risk`;
  if (dom.potentialCountVal) dom.potentialCountVal.textContent = potHot;

  // Metadata
  if (dom.metaResolution) dom.metaResolution.textContent = `${data.grid_size_m || 500}m Resolution`;
  if (dom.metaInferenceTime) {
    const infMs = data.execution_times ? data.execution_times.batch_inference_ms : null;
    dom.metaInferenceTime.textContent = infMs != null ? `Batch ML: ${infMs.toFixed(0)} ms` : "Batch ML: <5 ms";
  }
  if (dom.metaTotalLatency) {
    const totMs = data.execution_times ? data.execution_times.total_ms : null;
    dom.metaTotalLatency.textContent = totMs != null ? `Total Pipeline: ${totMs.toFixed(0)} ms` : "Total Pipeline: ~450 ms";
  }
}

function renderHotspotsTable(hotspots) {
  if (!dom.hotspotsTableBody) return;
  dom.hotspotsTableBody.innerHTML = "";

  if (!Array.isArray(hotspots) || hotspots.length === 0) {
    if (dom.hotspotsSection) dom.hotspotsSection.style.display = "none";
    return;
  }

  if (dom.hotspotsSection) dom.hotspotsSection.style.display = "block";

  hotspots.forEach(h => {
    const tr = document.createElement("tr");

    const tdLoc = document.createElement("td");
    tdLoc.innerHTML = `<strong>#${h.grid_id}</strong><div style="font-size: 10px; color: #94a3b8;">${h.lat.toFixed(4)}, ${h.lon.toFixed(4)}</div>`;
    tr.appendChild(tdLoc);

    const tdWl = document.createElement("td");
    tdWl.innerHTML = `<strong>${(h.waterlogging_probability * 100).toFixed(1)}%</strong> <span class="badge-pill ${getRiskBadgeClass(h.waterlogging_risk)}">${h.waterlogging_risk}</span>`;
    tr.appendChild(tdWl);

    const tdDr = document.createElement("td");
    tdDr.innerHTML = `<strong>${(h.drainage_probability * 100).toFixed(1)}%</strong> <span class="badge-pill ${getRiskBadgeClass(h.drainage_risk)}">${h.drainage_risk}</span>`;
    tr.appendChild(tdDr);

    const tdElev = document.createElement("td");
    tdElev.textContent = `${h.elevation != null ? h.elevation.toFixed(1) : '--'}m (${h.slope != null ? h.slope.toFixed(1) : '--'}°)`;
    tr.appendChild(tdElev);

    const tdStatus = document.createElement("td");
    const stBadge = h.combined_status === "CRITICAL_HOTSPOT"
      ? '<span class="badge-pill badge-danger">CRITICAL HOTSPOT</span>'
      : h.combined_status === "HIGH_RISK_HOTSPOT"
        ? '<span class="badge-pill badge-warning">HIGH RISK HOTSPOT</span>'
        : '<span class="badge-pill badge-info">POTENTIAL HOTSPOT</span>';
    tdStatus.innerHTML = `${stBadge}<div style="font-size: 9px; color: #94a3b8; margin-top: 2px;">Drainage/Waterlogging Risk Hotspot</div>`;
    tr.appendChild(tdStatus);

    const tdAct = document.createElement("td");
    const inspectBtn = document.createElement("button");
    inspectBtn.className = "btn-inspect-cell";
    inspectBtn.textContent = "Inspect";
    inspectBtn.addEventListener("click", () => {
      if (mapState.map) {
        mapState.map.setView([h.lat, h.lon], 15, { animate: true });
        mapState.wlGridLayer.eachLayer(layer => {
          if (layer.feature && layer.feature.properties && layer.feature.properties.grid_id === h.grid_id) {
            layer.openPopup();
          }
        });
      }
    });
    tdAct.appendChild(inspectBtn);
    tr.appendChild(tdAct);

    dom.hotspotsTableBody.appendChild(tr);
  });
}

// Global scope registration for popup click handlers
window.loadStateView = loadStateView;
window.loadDistrictView = loadDistrictView;
window.generateLocalRiskGrid = generateLocalRiskGrid;
// Close search dropdown on click outside
document.addEventListener("click", (e) => {
  if (!dom.placeSearchInput.contains(e.target) && !dom.placeSearchResults.contains(e.target) && !dom.searchLocationBtn.contains(e.target)) {
    dom.placeSearchResults.style.display = "none";
  }
});

// Analyze CTA Click Listener
dom.analyzeBtn.addEventListener("click", runAnalysis);

// =============================================================================
// 7. INITIAL PAGE LOAD
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadStatesCatalog();

  if (selectedLocation) {
    dom.selectedPlaceName.textContent = selectedLocation.name;
    dom.selectedLocationCard.style.display = "inline-flex";
    dom.placeSearchInput.value = selectedLocation.name;
  }

  // Initial analysis on page load
  runAnalysis();
});
