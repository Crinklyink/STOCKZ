import Foundation

struct LatestScan: Decodable {
    let selected: [Candidate]?
    let rawCandidates: [Candidate]?
    let displayCandidates: [Candidate]?
    let scanSummary: ScanSummary?
    let regimeLabel: String?
    let trainingReport: TrainingReport?
    let paperTradeSummary: PaperTradeSummary?
    let modelMonitoring: ModelMonitoring?
    let macro: MacroSnapshot?
    let macroSummary: String?
    let qualifiedCount: Int?
    let runtimeSeconds: Double?
    let generatedAt: String?
    let backtestRows: [BacktestRow]?

    enum CodingKeys: String, CodingKey {
        case selected
        case rawCandidates = "candidates"
        case displayCandidates = "display_candidates"
        case scanSummary = "scan_summary"
        case regimeLabel = "regime_label"
        case trainingReport = "training_report"
        case paperTradeSummary = "paper_trade_summary"
        case modelMonitoring = "model_monitoring"
        case macro
        case macroSummary = "macro_summary"
        case qualifiedCount = "qualified_count"
        case runtimeSeconds = "runtime_seconds"
        case generatedAt = "generated_at"
        case backtestRows = "backtest_rows"
    }

    var candidates: [Candidate] {
        if let selected, !selected.isEmpty { return selected }
        return displayCandidates ?? []
    }

    var allCandidates: [Candidate] {
        rawCandidates ?? displayCandidates ?? selected ?? []
    }

    var rejectedCandidates: [Candidate] {
        let picked = Set(candidates.map(\.ticker))
        return allCandidates
            .filter { !picked.contains($0.ticker) }
            .sorted {
                if ($0.marketCap ?? 0) != ($1.marketCap ?? 0) {
                    return ($0.marketCap ?? 0) > ($1.marketCap ?? 0)
                }
                return ($0.finalScore ?? 0) > ($1.finalScore ?? 0)
            }
    }
}

struct MacroSnapshot: Decodable {
    let riskRegime: String?
    let vix: Double?
    let topSectors: [String]?
    let bottomSectors: [String]?
    let sectorReturns: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case riskRegime = "risk_regime"
        case vix
        case topSectors = "top_sectors"
        case bottomSectors = "bottom_sectors"
        case sectorReturns = "sector_returns"
    }
}

struct PaperTradeSummary: Decodable {
    let weeks: Int?
    let targetHitRate: Double?
    let positiveReturnRate: Double?
    let averageReturn: Double?
    let bestPick: String?
    let bestReturn: Double?
    let worstPick: String?
    let worstReturn: Double?

    enum CodingKeys: String, CodingKey {
        case weeks
        case targetHitRate = "target_hit_rate"
        case positiveReturnRate = "positive_return_rate"
        case averageReturn = "average_return"
        case bestPick = "best_pick"
        case bestReturn = "best_return"
        case worstPick = "worst_pick"
        case worstReturn = "worst_return"
    }
}

struct Candidate: Decodable, Identifiable {
    var id: String { ticker }
    let ticker: String
    let companyName: String?
    let sector: String?
    let marketCap: Double?
    let finalScore: Double?
    let currentPrice: Double?
    let stopLoss: Double?
    let riskReward: Double?
    let kellySizePct: Double?
    let confluenceCount: Int?
    let probability4Pct5D: Double?
    let confidenceLabel: String?
    let mlScore: Double?
    let tierLabel: String?
    let aiExplanation: String?
    let notes: [String]?
    let targets: Targets?
    let diagnostics: CandidateDiagnostics?
    let technicalScore: Double?
    let volumeMomentumScore: Double?
    let sentimentScore: Double?
    let rsScore: Double?
    let institutionalScore: Double?
    let smartMoneyScore: Double?
    let sectorTailwindPoints: Double?
    let sectorTemperatureTag: String?
    let scoreLow: Double?
    let scoreHigh: Double?
    let scoreUncertainty: Double?
    let positionSizePct: Double?

    enum CodingKeys: String, CodingKey {
        case ticker
        case companyName = "company_name"
        case sector
        case marketCap = "market_cap"
        case finalScore = "final_score"
        case currentPrice = "current_price"
        case stopLoss = "stop_loss"
        case riskReward = "risk_reward"
        case kellySizePct = "kelly_size_pct"
        case confluenceCount = "confluence_count"
        case probability4Pct5D = "probability_4pct_5d"
        case confidenceLabel = "confidence_label"
        case mlScore = "ml_score"
        case tierLabel = "tier_label"
        case aiExplanation = "ai_explanation"
        case notes
        case targets
        case diagnostics
        case technicalScore = "technical_score"
        case volumeMomentumScore = "volume_momentum_score"
        case sentimentScore = "sentiment_score"
        case rsScore = "rs_score"
        case institutionalScore = "institutional_score"
        case smartMoneyScore = "smart_money_score"
        case sectorTailwindPoints = "sector_tailwind_points"
        case sectorTemperatureTag = "sector_temperature_tag"
        case scoreLow = "score_low"
        case scoreHigh = "score_high"
        case scoreUncertainty = "score_uncertainty"
        case positionSizePct = "position_size_pct"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // ticker is the only truly required field
        ticker          = try c.decode(String.self, forKey: .ticker)
        // All optional fields use try? so a type mismatch never crashes the whole array
        companyName     = try? c.decodeIfPresent(String.self,   forKey: .companyName)   ?? nil
        sector          = try? c.decodeIfPresent(String.self,   forKey: .sector)         ?? nil
        marketCap       = try? c.decodeIfPresent(Double.self, forKey: .marketCap) ?? nil
        tierLabel       = try? c.decodeIfPresent(String.self,   forKey: .tierLabel)      ?? nil
        aiExplanation   = try? c.decodeIfPresent(String.self,   forKey: .aiExplanation)  ?? nil
        finalScore      = try? c.decodeIfPresent(Double.self,   forKey: .finalScore)     ?? nil
        currentPrice    = try? c.decodeIfPresent(Double.self,   forKey: .currentPrice)   ?? nil
        stopLoss        = try? c.decodeIfPresent(Double.self,   forKey: .stopLoss)       ?? nil
        riskReward      = try? c.decodeIfPresent(Double.self,   forKey: .riskReward)     ?? nil
        kellySizePct    = try? c.decodeIfPresent(Double.self,   forKey: .kellySizePct)   ?? nil
        confluenceCount = try? c.decodeIfPresent(Int.self,      forKey: .confluenceCount) ?? nil
        probability4Pct5D = try? c.decodeIfPresent(Double.self, forKey: .probability4Pct5D) ?? nil
        confidenceLabel = try? c.decodeIfPresent(String.self, forKey: .confidenceLabel) ?? nil
        mlScore = try? c.decodeIfPresent(Double.self, forKey: .mlScore) ?? nil
        notes           = try? c.decodeIfPresent([String].self, forKey: .notes)          ?? nil
        targets         = try? c.decodeIfPresent(Targets.self,  forKey: .targets)        ?? nil
        diagnostics     = try? c.decodeIfPresent(CandidateDiagnostics.self, forKey: .diagnostics) ?? nil
        technicalScore  = try? c.decodeIfPresent(Double.self, forKey: .technicalScore) ?? nil
        volumeMomentumScore = try? c.decodeIfPresent(Double.self, forKey: .volumeMomentumScore) ?? nil
        sentimentScore  = try? c.decodeIfPresent(Double.self, forKey: .sentimentScore) ?? nil
        rsScore         = try? c.decodeIfPresent(Double.self, forKey: .rsScore) ?? nil
        institutionalScore = try? c.decodeIfPresent(Double.self, forKey: .institutionalScore) ?? nil
        smartMoneyScore = try? c.decodeIfPresent(Double.self, forKey: .smartMoneyScore) ?? nil
        sectorTailwindPoints = try? c.decodeIfPresent(Double.self, forKey: .sectorTailwindPoints) ?? nil
        sectorTemperatureTag = try? c.decodeIfPresent(String.self, forKey: .sectorTemperatureTag) ?? nil
        scoreLow        = try? c.decodeIfPresent(Double.self, forKey: .scoreLow) ?? nil
        scoreHigh       = try? c.decodeIfPresent(Double.self, forKey: .scoreHigh) ?? nil
        scoreUncertainty = try? c.decodeIfPresent(Double.self, forKey: .scoreUncertainty) ?? nil
        positionSizePct = try? c.decodeIfPresent(Double.self, forKey: .positionSizePct) ?? nil
    }

    var targetPrice: Double {
        targets?.tp2 ?? currentPrice ?? 0
    }

    var upsidePercent: Double {
        guard let currentPrice, currentPrice > 0 else { return 0 }
        return ((targetPrice / currentPrice) - 1) * 100
    }

    var stopDistancePercent: Double {
        guard let currentPrice, currentPrice > 0, let stopLoss else { return 0 }
        return ((stopLoss / currentPrice) - 1) * 100
    }

    var scoreBand: String {
        let score = finalScore ?? 0
        if score >= 70 { return "High conviction" }
        if score >= 54 { return "Qualified" }
        return "Speculative"
    }

    var confidencePercent: Double {
        probability4Pct5D ?? mlScore ?? finalScore ?? 0
    }

    var scoreBreakdown: [ScoreFactor] {
        [
            ScoreFactor(name: "Technicals", value: technicalScore ?? finalScore ?? 0, colorName: "blue"),
            ScoreFactor(name: "RS", value: rsScore ?? finalScore ?? 0, colorName: "green"),
            ScoreFactor(name: "Volume", value: volumeMomentumScore ?? finalScore ?? 0, colorName: "teal"),
            ScoreFactor(name: "ML Probability", value: probability4Pct5D ?? mlScore ?? finalScore ?? 0, colorName: "purple"),
            ScoreFactor(name: "Sentiment", value: sentimentScore ?? 50, colorName: "yellow"),
            ScoreFactor(name: "Institutional Flow", value: institutionalScore ?? smartMoneyScore ?? 50, colorName: "blue"),
            ScoreFactor(name: "Risk/Reward", value: min(max((riskReward ?? 0) / 3.0 * 100.0, 0), 100), colorName: "green")
        ]
    }
}

struct ScoreFactor: Identifiable {
    var id: String { name }
    let name: String
    let value: Double
    let colorName: String
}

struct Targets: Decodable {
    let tp1: Double?
    let tp2: Double?
    let tp3: Double?
}

struct CandidateDiagnostics: Decodable {
    let analyst: AnalystDiagnostics?
}

struct AnalystDiagnostics: Decodable {
    let whyThisPick: [String]?
    let negativeDrivers: [String]?
    let invalidation: [String]?
    let similarHistoricalSetups: [String]?
    let backtestConfidence: Double?
    let modelConfidence: String?
    let riskRewardMap: RiskRewardMap?
    let dataFreshness: String?
    let dataQualityWarnings: [String]?
    let whyNotOfficial: String?

    enum CodingKeys: String, CodingKey {
        case whyThisPick = "why_this_pick"
        case negativeDrivers = "negative_drivers"
        case invalidation
        case similarHistoricalSetups = "similar_historical_setups"
        case backtestConfidence = "backtest_confidence"
        case modelConfidence = "model_confidence"
        case riskRewardMap = "risk_reward_map"
        case dataFreshness = "data_freshness"
        case dataQualityWarnings = "data_quality_warnings"
        case whyNotOfficial = "why_not_official"
    }
}

struct RiskRewardMap: Decodable {
    let entry: Double?
    let stop: Double?
    let target: Double?
    let rewardPct: Double?
    let riskPct: Double?
    let riskReward: Double?

    enum CodingKeys: String, CodingKey {
        case entry
        case stop
        case target
        case rewardPct = "reward_pct"
        case riskPct = "risk_pct"
        case riskReward = "risk_reward"
    }
}

struct ModelMonitoring: Decodable {
    let health: String?
    let modelAuc: Double?
    let candidateCount: Int?
    let sectorConcentration: SectorConcentration?
    let warnings: [String]?
    let scoreDistribution: ScoreDistribution?
    let calibration: CalibrationReport?

    enum CodingKeys: String, CodingKey {
        case health
        case modelAuc = "model_auc"
        case candidateCount = "candidate_count"
        case sectorConcentration = "sector_concentration"
        case warnings
        case scoreDistribution = "score_distribution"
        case calibration
    }
}

struct ScoreDistribution: Decodable {
    let mean: Double?
    let std: Double?
}

struct CalibrationReport: Decodable {
    let buckets: [CalibrationBucket]?
    let warning: String?
}

struct CalibrationBucket: Decodable, Identifiable {
    var id: String { bucket ?? UUID().uuidString }
    let bucket: String?
    let trades: Int?
    let hitRate: Double?
    let avgReturn: Double?

    enum CodingKeys: String, CodingKey {
        case bucket
        case trades
        case hitRate = "hit_rate"
        case avgReturn = "avg_return"
    }
}

struct SectorConcentration: Decodable {
    let topSector: String?
    let topSectorShare: Double?
    let warning: String?

    enum CodingKeys: String, CodingKey {
        case topSector = "top_sector"
        case topSectorShare = "top_sector_share"
        case warning
    }
}

struct ScanSummary: Decodable {
    let vix: Double?
    let regime: String?
    let spyWeekReturn: Double?
    let selectionWarning: String?

    enum CodingKeys: String, CodingKey {
        case vix
        case regime
        case spyWeekReturn = "spy_week_return"
        case selectionWarning = "selection_warning"
    }
}

struct BacktestRow: Decodable, Identifiable {
    var id: String { "\(runID)-\(ticker)" }
    let runID: String
    let createdAt: String?
    let ticker: String
    let finalScore: Double?
    let realizedReturn: Double?
    let resolvedTargetHit: Double?
    let confidenceLabel: String?

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case createdAt = "created_at"
        case ticker
        case finalScore = "final_score"
        case realizedReturn = "realized_return"
        case resolvedTargetHit = "resolved_target_hit"
        case confidenceLabel = "confidence_label"
    }
}

struct TrainingReport: Decodable {
    let trained: Bool?
    let trainedAt: String?
    let trainingSamples: Int?
    let validationSamples: Int?
    let modelFamily: String?
    let selectedProfile: String?
    let labelDefinition: String?
    let auc: Double?
    let ensembleAuc: Double?
    let xgbAuc: Double?
    let lightgbmAuc: Double?
    let shortHorizonAuc: Double?
    let featureImportance: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case trained
        case trainedAt = "trained_at"
        case trainingSamples = "training_samples"
        case validationSamples = "validation_samples"
        case modelFamily = "model_family"
        case selectedProfile = "selected_profile"
        case labelDefinition = "label_definition"
        case auc
        case ensembleAuc = "ensemble_auc"
        case xgbAuc = "xgb_auc"
        case lightgbmAuc = "lightgbm_auc"
        case shortHorizonAuc = "short_horizon_auc"
        case featureImportance = "feature_importance"
    }
}

struct ModelMetadata: Decodable {
    let trained: Bool?
    let trainedAt: String?
    let trainingSamples: Int?
    let validationSamples: Int?
    let modelFamily: String?
    let selectedProfile: String?
    let auc: Double?
    let ensembleAuc: Double?
    let xgbAuc: Double?
    let lightgbmAuc: Double?
    let shortHorizonAuc: Double?
    let labelDefinition: String?
    let ensembleWeights: [String: Double]?
    let featureImportance: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case trained
        case trainedAt = "trained_at"
        case trainingSamples = "training_samples"
        case validationSamples = "validation_samples"
        case modelFamily = "model_family"
        case selectedProfile = "selected_profile"
        case auc
        case ensembleAuc = "ensemble_auc"
        case xgbAuc = "xgb_auc"
        case lightgbmAuc = "lightgbm_auc"
        case shortHorizonAuc = "short_horizon_auc"
        case labelDefinition = "label_definition"
        case ensembleWeights = "ensemble_weights"
        case featureImportance = "feature_importance"
    }

    var activeAuc: Double {
        ensembleAuc ?? auc ?? 0
    }

    var stackName: String {
        if modelFamily == "RandomForestFallback" { return "Random Forest" }
        if let lightgbmAuc, lightgbmAuc > 0.5 { return "XGBoost + LightGBM" }
        return modelFamily ?? "Adaptive Ensemble"
    }

    var targetSummary: String {
        if labelDefinition?.contains("+6%") == true { return "+6% / -3.5% weekly barrier" }
        return "weekly target"
    }

    var trainedDateText: String {
        trainedAt?.prefix(10).description ?? "not trained"
    }

    var modeSummary: String {
        "\(stackName) | \(targetSummary)"
    }
}

struct TrainingEvent: Identifiable {
    let id = UUID()
    let date = Date()
    let text: String
}

enum ScanStageStatus {
    case pending
    case active
    case complete
}

struct ScanStage: Identifiable {
    let id: String
    let order: Int
    let title: String
    let symbol: String
    var detail: String
    var status: ScanStageStatus = .pending

    static let defaults: [ScanStage] = [
        ScanStage(id: "fetching_data", order: 1, title: "Fetching data", symbol: "tray.and.arrow.down", detail: "Market, macro, price, volume"),
        ScanStage(id: "scoring", order: 2, title: "Scoring", symbol: "function", detail: "Technicals, RS, ML, sentiment"),
        ScanStage(id: "filtering", order: 3, title: "Filtering", symbol: "line.3.horizontal.decrease", detail: "Liquidity, risk, data quality"),
        ScanStage(id: "ranking", order: 4, title: "Ranking", symbol: "list.number", detail: "Score, confidence, risk/reward"),
        ScanStage(id: "saving", order: 5, title: "Saving artifacts", symbol: "square.and.arrow.down", detail: "Reports, watchlists, history")
    ]
}

enum AppSection: String, CaseIterable, Identifiable {
    case dashboard = "Dashboard"
    case picks = "Picks"
    case risk = "Risk"
    case alerts = "Alerts"
    case training = "Training"
    case backtesting = "Backtesting"
    case modelLab = "Model Lab"
    case history = "History"
    case settings = "Settings"

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .dashboard: "chart.xyaxis.line"
        case .picks: "scope"
        case .risk: "exclamationmark.shield"
        case .alerts: "bell.badge"
        case .training: "cpu"
        case .backtesting: "clock.arrow.circlepath"
        case .modelLab: "point.3.connected.trianglepath.dotted"
        case .history: "list.bullet.rectangle"
        case .settings: "gearshape"
        }
    }
}
