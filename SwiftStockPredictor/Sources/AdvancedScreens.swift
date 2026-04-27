import SwiftUI

struct RiskDashboardScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        PageFrame {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Risk Dashboard", subtitle: "Concentration, volatility regime, stop distance, and sizing warnings.")
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 12)], spacing: 12) {
                    MetricTile(title: "VIX Regime", value: String(format: "%.1f", appState.summary?.vix ?? 0), footnote: riskRegime, tone: vixTone)
                    MetricTile(title: "Top Sector", value: topSector, footnote: topSectorShare, tone: sectorTone)
                    MetricTile(title: "Avg Stop", value: averageStop.percentText, footnote: "distance", tone: averageStop < -7 ? Design.red : Design.yellow)
                    MetricTile(title: "Size Warning", value: shouldReduceSize ? "Reduce" : "Normal", footnote: warningText, tone: shouldReduceSize ? Design.red : Design.green)
                }
                GlassPanel {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Sector Crowding")
                            .font(.headline)
                        ForEach(sectorRows, id: \.sector) { row in
                            HStack {
                                Text(row.sector)
                                    .frame(width: 170, alignment: .leading)
                                MiniBar(value: row.share / 100.0, tint: row.share >= 50 ? Design.red : Design.blue)
                                Text(row.share.percentText)
                                    .font(.callout.monospacedDigit())
                                    .frame(width: 64, alignment: .trailing)
                            }
                        }
                    }
                }
                GlassPanel {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Warnings")
                            .font(.headline)
                        BulletList(items: riskWarnings)
                    }
                }
            }
        }
    }

    private var riskRegime: String {
        let vix = appState.summary?.vix ?? 0
        if vix >= 25 { return "high volatility" }
        if vix >= 18 { return "elevated" }
        return "calm"
    }

    private var vixTone: Color {
        (appState.summary?.vix ?? 0) >= 25 ? Design.red : ((appState.summary?.vix ?? 0) >= 18 ? Design.yellow : Design.green)
    }

    private var sectorRows: [(sector: String, share: Double)] {
        let picks = appState.candidates
        guard !picks.isEmpty else { return [] }
        let grouped = Dictionary(grouping: picks, by: { $0.sector ?? "Unknown" })
        return grouped.map { ($0.key, Double($0.value.count) / Double(picks.count) * 100) }
            .sorted { $0.share > $1.share }
    }

    private var topSector: String {
        appState.latestScan?.modelMonitoring?.sectorConcentration?.topSector ?? sectorRows.first?.sector ?? "--"
    }

    private var topSectorShare: String {
        if let share = appState.latestScan?.modelMonitoring?.sectorConcentration?.topSectorShare {
            return share.percentText
        }
        return (sectorRows.first?.share ?? 0).percentText
    }

    private var sectorTone: Color {
        ((appState.latestScan?.modelMonitoring?.sectorConcentration?.topSectorShare ?? sectorRows.first?.share ?? 0) >= 50) ? Design.red : Design.green
    }

    private var averageStop: Double {
        let stops = appState.candidates.map(\.stopDistancePercent).filter { $0 != 0 }
        guard !stops.isEmpty else { return 0 }
        return stops.reduce(0, +) / Double(stops.count)
    }

    private var shouldReduceSize: Bool {
        (appState.summary?.vix ?? 0) >= 25 || sectorTone == Design.red || averageStop < -8
    }

    private var warningText: String {
        shouldReduceSize ? "risk stacked" : "risk balanced"
    }

    private var riskWarnings: [String] {
        var warnings = appState.latestScan?.modelMonitoring?.warnings ?? []
        if (appState.summary?.vix ?? 0) >= 25 { warnings.append("VIX is high enough to reduce paper position size.") }
        if sectorTone == Design.red { warnings.append("Top sector concentration is elevated.") }
        if averageStop < -8 { warnings.append("Average stop distance is wide; size smaller.") }
        return warnings.isEmpty ? ["No major concentration or volatility warnings."] : warnings
    }
}

struct AlertsScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        PageFrame {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Alerts", subtitle: "Target, stop, rank, and score-change rules for active picks.")
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 12)], spacing: 12) {
                    MetricTile(title: "Triggered", value: "\(alerts.filter { $0.state == "triggered" }.count)", footnote: "needs action", tone: Design.red)
                    MetricTile(title: "Near", value: "\(alerts.filter { $0.state == "near" }.count)", footnote: "within 2%", tone: Design.yellow)
                    MetricTile(title: "Armed", value: "\(alerts.filter { $0.state == "armed" }.count)", footnote: "watching")
                    MetricTile(title: "Top 10", value: "\(appState.candidates.count)", footnote: "rank alerts")
                }
                GlassPanel {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(alerts) { alert in
                            HStack(spacing: 12) {
                                Image(systemName: alert.symbol)
                                    .foregroundStyle(alert.tone)
                                    .frame(width: 28, height: 28)
                                    .background(alert.tone.opacity(0.14), in: RoundedRectangle(cornerRadius: Design.radius))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(alert.title)
                                        .font(.callout.weight(.semibold))
                                    Text(alert.detail)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Pill(text: alert.state, color: alert.tone)
                            }
                            Divider().opacity(0.35)
                        }
                    }
                }
            }
        }
    }

    private var alerts: [PickAlert] {
        var rows = appState.candidates.prefix(10).flatMap { candidate in
            alertRules(for: candidate)
        }
        rows.append(contentsOf: rankMovementAlerts)
        return rows.sorted { lhs, rhs in
            priority(lhs.state) < priority(rhs.state)
        }
    }

    private func alertRules(for candidate: Candidate) -> [PickAlert] {
        let price = candidate.currentPrice ?? 0
        let target = candidate.targetPrice
        let stop = candidate.stopLoss ?? 0
        let targetDistance = target > 0 ? ((target / max(price, 0.01)) - 1) * 100.0 : 0
        let stopDistance = stop > 0 ? ((price / stop) - 1) * 100.0 : 0
        let targetState = price >= target && target > 0 ? "triggered" : (targetDistance <= 2 ? "near" : "armed")
        let stopState = price <= stop && stop > 0 ? "triggered" : (stopDistance <= 2 ? "near" : "armed")
        let scoreState = (candidate.scoreHigh ?? candidate.finalScore ?? 0) >= 70 ? "near" : "armed"
        return [
            PickAlert(title: "\(candidate.ticker) target", detail: "Target \(target.moneyText); current \(price.moneyText); \(targetDistance.percentText) away", state: targetState, symbol: "target", tone: targetState == "triggered" ? Design.green : Design.blue),
            PickAlert(title: "\(candidate.ticker) stop", detail: "Stop \(stop.moneyText); current \(price.moneyText); \(stopDistance.percentText) cushion", state: stopState, symbol: "shield", tone: stopState == "triggered" ? Design.red : Design.yellow),
            PickAlert(title: "\(candidate.ticker) score improves", detail: "Watch for score above \(String(format: "%.1f", (candidate.finalScore ?? 0) + 5)) or high-conviction band.", state: scoreState, symbol: "arrow.up.right", tone: Design.purple)
        ]
    }

    private var rankMovementAlerts: [PickAlert] {
        let picked = Set(appState.candidates.map(\.ticker))
        return (appState.latestScan?.allCandidates ?? [])
            .filter { !picked.contains($0.ticker) }
            .sorted { ($0.finalScore ?? 0) > ($1.finalScore ?? 0) }
            .prefix(4)
            .map { candidate in
                PickAlert(
                    title: "\(candidate.ticker) nearing top 10",
                    detail: "Score \(String(format: "%.1f", candidate.finalScore ?? 0)); \(candidate.diagnostics?.analyst?.whyNotOfficial ?? "outside current rank list")",
                    state: (candidate.finalScore ?? 0) >= 50 ? "near" : "armed",
                    symbol: "arrow.up.arrow.down",
                    tone: Design.teal
                )
            }
    }

    private func priority(_ state: String) -> Int {
        switch state {
        case "triggered": 0
        case "near": 1
        default: 2
        }
    }
}

struct PickAlert: Identifiable {
    let id = UUID()
    let title: String
    let detail: String
    let state: String
    let symbol: String
    let tone: Color
}

struct ModelLabScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        PageFrame {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Model Lab", subtitle: "AUC, feature importance, samples, label target, recent tests, and drift checks.")
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
                    MetricTile(title: "AUC", value: String(format: "%.3f", appState.modelMetadata?.activeAuc ?? 0), footnote: "recent validation", tone: Design.green)
                    MetricTile(title: "Samples", value: (appState.modelMetadata?.trainingSamples ?? 0).compactText, footnote: "training")
                    MetricTile(title: "Validation", value: (appState.modelMetadata?.validationSamples ?? 0).compactText, footnote: "holdout")
                    MetricTile(title: "Target", value: "+6%", footnote: appState.modelMetadata?.targetSummary ?? "label")
                    MetricTile(title: "Drift", value: driftState, footnote: driftDetail, tone: driftTone)
                }
                HStack(alignment: .top, spacing: 14) {
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Feature Importance")
                                .font(.headline)
                            ForEach(featureRows.prefix(12), id: \.name) { row in
                                HStack {
                                    Text(row.name.cleanedProfile)
                                        .font(.caption)
                                        .frame(width: 170, alignment: .leading)
                                    MiniBar(value: row.value / maxFeature, tint: Design.blue)
                                    Text(String(format: "%.3f", row.value))
                                        .font(.caption.monospacedDigit())
                                        .frame(width: 54, alignment: .trailing)
                                }
                            }
                        }
                    }
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Recent Test Performance")
                                .font(.headline)
                            ForEach(appState.latestScan?.modelMonitoring?.calibration?.buckets ?? []) { bucket in
                                DataRow(
                                    title: bucket.bucket ?? "--",
                                    value: String(format: "%.0f%% hit | %@ avg", bucket.hitRate ?? 0, (bucket.avgReturn ?? 0).percentText),
                                    tone: (bucket.hitRate ?? 0) >= 70 ? Design.green : Design.yellow
                                )
                            }
                            Divider()
                            DataRow(title: "Stack", value: appState.modelMetadata?.stackName ?? "--", tone: Design.blue)
                            DataRow(title: "Profile", value: appState.modelMetadata?.selectedProfile?.cleanedProfile ?? "--")
                            DataRow(title: "Trained", value: appState.modelMetadata?.trainedDateText ?? "--")
                        }
                    }
                    .frame(width: 390)
                }
            }
        }
    }

    private var featureRows: [(name: String, value: Double)] {
        (appState.modelMetadata?.featureImportance ?? appState.latestScan?.trainingReport?.featureImportance ?? [:])
            .map { ($0.key, $0.value) }
            .sorted { $0.value > $1.value }
    }

    private var maxFeature: Double {
        max(featureRows.first?.value ?? 1, 0.0001)
    }

    private var driftState: String {
        (appState.latestScan?.modelMonitoring?.warnings ?? []).isEmpty ? "Stable" : "Watch"
    }

    private var driftDetail: String {
        appState.latestScan?.modelMonitoring?.warnings?.first ?? "no warning"
    }

    private var driftTone: Color {
        driftState == "Stable" ? Design.green : Design.yellow
    }
}

extension Double {
    var percentFreeText: String {
        String(format: "%.1f", self)
    }
}
