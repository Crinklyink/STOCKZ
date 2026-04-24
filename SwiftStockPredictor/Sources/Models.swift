import Foundation

struct LatestScan: Decodable {
    let selected: [Candidate]?
    let displayCandidates: [Candidate]?
    let scanSummary: ScanSummary?
    let regimeLabel: String?
    let trainingReport: TrainingReport?
    let paperTradeSummary: PaperTradeSummary?
    let macroSummary: String?
    let qualifiedCount: Int?
    let runtimeSeconds: Double?
    let generatedAt: String?

    enum CodingKeys: String, CodingKey {
        case selected
        case displayCandidates = "display_candidates"
        case scanSummary = "scan_summary"
        case regimeLabel = "regime_label"
        case trainingReport = "training_report"
        case paperTradeSummary = "paper_trade_summary"
        case macroSummary = "macro_summary"
        case qualifiedCount = "qualified_count"
        case runtimeSeconds = "runtime_seconds"
        case generatedAt = "generated_at"
    }

    var candidates: [Candidate] {
        if let selected, !selected.isEmpty { return selected }
        return displayCandidates ?? []
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
    let finalScore: Double?
    let currentPrice: Double?
    let stopLoss: Double?
    let riskReward: Double?
    let kellySizePct: Double?
    let confluenceCount: Int?
    let tierLabel: String?
    let aiExplanation: String?
    let notes: [String]?
    let targets: Targets?

    enum CodingKeys: String, CodingKey {
        case ticker
        case companyName = "company_name"
        case sector
        case finalScore = "final_score"
        case currentPrice = "current_price"
        case stopLoss = "stop_loss"
        case riskReward = "risk_reward"
        case kellySizePct = "kelly_size_pct"
        case confluenceCount = "confluence_count"
        case tierLabel = "tier_label"
        case aiExplanation = "ai_explanation"
        case notes
        case targets
    }

    var targetPrice: Double {
        targets?.tp2 ?? currentPrice ?? 0
    }

    var upsidePercent: Double {
        guard let currentPrice, currentPrice > 0 else { return 0 }
        return ((targetPrice / currentPrice) - 1) * 100
    }
}

struct Targets: Decodable {
    let tp1: Double?
    let tp2: Double?
    let tp3: Double?
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

struct TrainingReport: Decodable {
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
        case ensembleWeights = "ensemble_weights"
        case featureImportance = "feature_importance"
    }

    var activeAuc: Double {
        ensembleAuc ?? auc ?? 0
    }

    var stackName: String {
        if lightgbmAuc != nil { return "XGBoost + LightGBM" }
        return modelFamily ?? "Adaptive Ensemble"
    }
}

struct TrainingEvent: Identifiable {
    let id = UUID()
    let date = Date()
    let text: String
}

enum AppSection: String, CaseIterable, Identifiable {
    case dashboard = "Dashboard"
    case picks = "Picks"
    case training = "Training"
    case backtesting = "Backtesting"
    case history = "History"
    case settings = "Settings"

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .dashboard: "chart.xyaxis.line"
        case .picks: "scope"
        case .training: "cpu"
        case .backtesting: "clock.arrow.circlepath"
        case .history: "list.bullet.rectangle"
        case .settings: "gearshape"
        }
    }
}
