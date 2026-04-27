import SwiftUI
import AppKit

enum PicksMode: String, CaseIterable, Identifiable {
    case ranked = "Ranked"
    case compare = "Compare"
    case portfolio = "Portfolio"
    case whyNot = "Why Not"

    var id: String { rawValue }
}

struct PicksScreen: View {
    @EnvironmentObject private var appState: AppState
    @State private var mode: PicksMode = .ranked
    @State private var detailCandidate: Candidate?

    var body: some View {
        PageFrame {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Picks", subtitle: "Ranked candidates with clean risk, score, and model context.")
                StatusStrip(
                    title: picksStatusTitle,
                    detail: picksStatusDetail,
                    symbol: appState.scanUsesOlderModel ? "arrow.triangle.2.circlepath" : "scope",
                    color: appState.scanUsesOlderModel ? Design.yellow : (appState.candidates.isEmpty ? .secondary : Design.blue)
                )

                Picker("Picks mode", selection: $mode) {
                    ForEach(PicksMode.allCases) { item in
                        Text(item.rawValue).tag(item)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 520)

                if appState.candidates.isEmpty {
                    GlassPanel {
                        EmptyState(text: "No current candidates. Run Scan from the sidebar to generate this week's list.")
                    }
                } else {
                    switch mode {
                    case .ranked:
                        LazyVStack(spacing: 10) {
                            ForEach(Array(appState.candidates.prefix(24).enumerated()), id: \.element.id) { index, candidate in
                                Button {
                                    detailCandidate = candidate
                                } label: {
                                    PickCard(candidate: candidate, rank: index + 1)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    case .compare:
                        WatchlistCompareView(candidates: Array(appState.candidates.prefix(10))) { candidate in
                            detailCandidate = candidate
                        }
                    case .portfolio:
                        PortfolioSimulatorView(candidates: Array(appState.candidates.prefix(10))) { candidate in
                            detailCandidate = candidate
                        }
                    case .whyNot:
                        WhyNotPickedView(candidates: Array(appState.latestScan?.rejectedCandidates.prefix(14) ?? [])) { candidate in
                            detailCandidate = candidate
                        }
                    }
                }
            }
        }
        .sheet(item: $detailCandidate) { candidate in
            PickDetailDrilldown(candidate: candidate)
                .environmentObject(appState)
                .frame(minWidth: 760, minHeight: 720)
        }
    }

    private var picksStatusTitle: String {
        if appState.scanUsesOlderModel { return "Watchlist is from the previous model" }
        return appState.candidates.isEmpty ? "No active watchlist" : "\(appState.candidates.count) candidates ready"
    }

    private var picksStatusDetail: String {
        if appState.scanUsesOlderModel {
            return "Run Scan to regenerate candidates with \(appState.activeModelSummary)."
        }
        return appState.latestScan?.macroSummary ?? "Run a scan to refresh the current market context."
    }
}

struct PickDetailDrilldown: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss
    let candidate: Candidate

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(candidate.ticker)
                            .font(.system(size: 36, weight: .semibold, design: .rounded))
                        Text(candidate.companyName ?? candidate.sector ?? "Unknown company")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                    .font(.title3)
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
                    MetricTile(title: "Score", value: String(format: "%.1f", candidate.finalScore ?? 0), footnote: candidate.scoreBand, tone: scoreColor)
                    MetricTile(title: "Confidence", value: String(format: "%.0f%%", candidate.confidencePercent), footnote: candidate.confidenceLabel ?? "model")
                    MetricTile(title: "Target", value: candidate.targetPrice.moneyText, footnote: "\(candidate.upsidePercent.percentText) upside", tone: Design.green)
                    MetricTile(title: "Stop", value: (candidate.stopLoss ?? 0).moneyText, footnote: candidate.stopDistancePercent.percentText, tone: Design.red)
                }

                HStack(alignment: .top, spacing: 14) {
                    DetailChartPanel(candidate: candidate)
                        .frame(minWidth: 300)
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Why It Ranked")
                                .font(.headline)
                            BulletList(items: candidate.diagnostics?.analyst?.whyThisPick ?? candidate.notes ?? [])
                            Divider()
                            DataRow(title: "Sector", value: candidate.sector ?? "Unknown", tone: Design.blue)
                            DataRow(title: "Sector strength", value: sectorStrengthText, tone: sectorStrengthTone)
                            DataRow(title: "Model confidence", value: modelConfidenceText, tone: Design.purple)
                            if let low = candidate.scoreLow, let high = candidate.scoreHigh {
                                DataRow(title: "Score range", value: "\(String(format: "%.1f", low)) - \(String(format: "%.1f", high))")
                            }
                        }
                    }
                    .frame(maxWidth: 320)
                }

                GlassPanel {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Interactive Score Breakdown")
                            .font(.headline)
                        ScoreBreakdownView(candidate: candidate)
                    }
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 12)], spacing: 12) {
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Stop / Target")
                                .font(.headline)
                            RiskCell(title: "Entry", value: (candidate.currentPrice ?? 0).moneyText)
                            RiskCell(title: "Stop", value: (candidate.stopLoss ?? 0).moneyText)
                            RiskCell(title: "Target 1", value: candidate.targets?.tp1?.moneyText ?? "--")
                            RiskCell(title: "Target 2", value: candidate.targets?.tp2?.moneyText ?? "--")
                            RiskCell(title: "Target 3", value: candidate.targets?.tp3?.moneyText ?? "--")
                        }
                    }
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("Invalidation Rules")
                                .font(.headline)
                            BulletList(items: candidate.diagnostics?.analyst?.invalidation ?? fallbackInvalidation)
                        }
                    }
                }
            }
            .padding(22)
        }
    }

    private var scoreColor: Color {
        (candidate.finalScore ?? 0) >= 65 ? Design.green : Design.yellow
    }

    private var sectorStrengthText: String {
        guard let sector = candidate.sector else { return "Unknown" }
        if let value = appState.latestScan?.macro?.sectorReturns?[sector] {
            return "\(sector) \(String(format: "%+.1f%%", value * 100.0))"
        }
        return candidate.sectorTemperatureTag ?? (appState.latestScan?.macro?.topSectors?.contains(sector) == true ? "Macro tailwind" : "Neutral")
    }

    private var sectorStrengthTone: Color {
        guard let sector = candidate.sector else { return .secondary }
        let value = appState.latestScan?.macro?.sectorReturns?[sector] ?? 0
        if value > 0.02 { return Design.green }
        if value < -0.02 { return Design.red }
        return Design.teal
    }

    private var modelConfidenceText: String {
        if let confidence = candidate.diagnostics?.analyst?.modelConfidence { return confidence }
        let uncertainty = candidate.scoreUncertainty.map { String(format: " +/- %.1f", $0) } ?? ""
        return "\(candidate.confidenceLabel ?? "model") \(String(format: "%.0f%%", candidate.confidencePercent))\(uncertainty)"
    }

    private var fallbackInvalidation: [String] {
        [
            "Breaks below stop at \((candidate.stopLoss ?? 0).moneyText).",
            "Relative strength fades versus sector leaders.",
            "Volume dries up before price clears the next target."
        ]
    }
}

struct DetailChartPanel: View {
    @EnvironmentObject private var appState: AppState
    let candidate: Candidate

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 12) {
                Text("Chart")
                    .font(.headline)
                if let image = chartImage {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
                } else {
                    SyntheticChart(candidate: candidate)
                        .frame(height: 220)
                }
            }
        }
    }

    private var chartImage: NSImage? {
        let path = appState.projectRoot.appendingPathComponent("reports/\(candidate.ticker)_chart.png").path
        return NSImage(contentsOfFile: path)
    }
}

struct SyntheticChart: View {
    let candidate: Candidate

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .bottomLeading) {
                RoundedRectangle(cornerRadius: Design.radius).fill(.secondary.opacity(0.08))
                Path { path in
                    let points = samples
                    let step = proxy.size.width / CGFloat(max(points.count - 1, 1))
                    for index in points.indices {
                        let point = CGPoint(x: CGFloat(index) * step, y: proxy.size.height * (1 - points[index]))
                        index == 0 ? path.move(to: point) : path.addLine(to: point)
                    }
                }
                .stroke(Design.green, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
            }
        }
    }

    private var samples: [Double] {
        let base = min(max((candidate.finalScore ?? 50) / 100, 0.25), 0.85)
        return [0.26, 0.30, 0.25, 0.38, 0.34, 0.45, 0.48, 0.58, base, min(base + 0.08, 0.94)]
    }
}

struct ScoreBreakdownView: View {
    let candidate: Candidate
    @State private var selectedFactor: ScoreFactor?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(candidate.scoreBreakdown) { factor in
                Button {
                    selectedFactor = factor
                } label: {
                    HStack(spacing: 10) {
                        Text(factor.name)
                            .font(.callout)
                            .frame(width: 150, alignment: .leading)
                        MiniBar(value: factor.value / 100.0, tint: color(for: factor.colorName))
                        Text(String(format: "%.0f", factor.value))
                            .font(.callout.monospacedDigit().weight(.semibold))
                            .frame(width: 42, alignment: .trailing)
                    }
                }
                .buttonStyle(.plain)
                .padding(6)
                .background((selectedFactor?.id == factor.id ? color(for: factor.colorName).opacity(0.10) : .clear), in: RoundedRectangle(cornerRadius: Design.radius))
            }
            if let selectedFactor {
                Divider().opacity(0.35)
                Text(factorDetail(selectedFactor))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func color(for name: String) -> Color {
        switch name {
        case "green": Design.green
        case "teal": Design.teal
        case "purple": Design.purple
        case "yellow": Design.yellow
        default: Design.blue
        }
    }

    private func factorDetail(_ factor: ScoreFactor) -> String {
        switch factor.name {
        case "Technicals": "Trend, breakout, moving-average, volatility, and pattern inputs."
        case "RS": "Relative strength versus the market and peer group across multiple windows."
        case "Volume": "Volume expansion, momentum confirmation, and accumulation behavior."
        case "ML Probability": "Model estimate for the short-horizon target event."
        case "Sentiment": "News and social tone where available, neutralized when optional data is missing."
        case "Institutional Flow": "Institutional and smart-money proxies, including accumulation and divergence signals."
        default: "Reward to stop distance, normalized to the model's preferred range."
        }
    }
}

struct WatchlistCompareView: View {
    let candidates: [Candidate]
    let onOpen: (Candidate) -> Void
    @State private var selected: Set<String> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Watchlist Compare")
                        .font(.headline)
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), spacing: 8)], spacing: 8) {
                        ForEach(candidates) { candidate in
                            Toggle(candidate.ticker, isOn: Binding(
                                get: { selected.contains(candidate.ticker) },
                                set: { isOn in
                                    if isOn {
                                        if selected.count < 4 { selected.insert(candidate.ticker) }
                                    } else {
                                        selected.remove(candidate.ticker)
                                    }
                                }
                            ))
                            .toggleStyle(.button)
                        }
                    }
                    Text("Select 2-4 stocks.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 12)], spacing: 12) {
                ForEach(compareCandidates) { candidate in
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 10) {
                            Button {
                                onOpen(candidate)
                            } label: {
                                Label(candidate.ticker, systemImage: "arrow.up.right.square")
                                    .font(.title3.weight(.semibold))
                            }
                            .buttonStyle(.plain)
                            DataRow(title: "Score", value: String(format: "%.1f", candidate.finalScore ?? 0), tone: Design.blue)
                            DataRow(title: "Upside", value: candidate.upsidePercent.percentText, tone: Design.green)
                            DataRow(title: "Risk", value: candidate.stopDistancePercent.percentText, tone: Design.red)
                            DataRow(title: "Sector", value: candidate.sector ?? "--")
                            DataRow(title: "Probability", value: String(format: "%.0f%%", candidate.confidencePercent), tone: Design.purple)
                            DataRow(title: "Momentum", value: String(format: "%.0f", candidate.volumeMomentumScore ?? candidate.technicalScore ?? 0), tone: Design.teal)
                        }
                    }
                }
            }
        }
        .onAppear {
            if selected.isEmpty {
                selected = Set(candidates.prefix(3).map(\.ticker))
            }
        }
    }

    private var compareCandidates: [Candidate] {
        candidates.filter { selected.contains($0.ticker) }
    }
}

struct PortfolioSimulatorView: View {
    let candidates: [Candidate]
    let onOpen: (Candidate) -> Void
    @State private var accountSize: Double = 100000
    @State private var allocations: [String: Double] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            GlassPanel {
                HStack {
                    Text("Paper Portfolio")
                        .font(.headline)
                    Spacer()
                    TextField("Account", value: $accountSize, format: .currency(code: "USD"))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 150)
                }
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 12)], spacing: 12) {
                MetricTile(title: "Exposure", value: totalAllocation.percentText, footnote: "allocated")
                MetricTile(title: "Expected Upside", value: expectedUpside.moneyText, footnote: "at targets", tone: Design.green)
                MetricTile(title: "Stop Risk", value: stopRisk.moneyText, footnote: "paper loss", tone: Design.red)
                MetricTile(title: "Max Loss", value: stopRisk.moneyText, footnote: maxLossWarning, tone: stopRisk / max(accountSize, 1) > 0.08 ? Design.red : Design.yellow)
            }
            GlassPanel {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(candidates) { candidate in
                        HStack {
                            Button(candidate.ticker) {
                                onOpen(candidate)
                            }
                            .buttonStyle(.plain)
                            .font(.callout.weight(.semibold))
                            .frame(width: 58, alignment: .leading)
                            Slider(value: Binding(
                                get: { allocations[candidate.ticker, default: candidate.positionSizePct ?? candidate.kellySizePct ?? 0] },
                                set: { allocations[candidate.ticker] = $0 }
                            ), in: 0...20, step: 0.5)
                            Text(String(format: "%.1f%%", allocations[candidate.ticker, default: candidate.positionSizePct ?? candidate.kellySizePct ?? 0]))
                                .font(.callout.monospacedDigit())
                                .frame(width: 58, alignment: .trailing)
                            Text((accountSize * allocation(for: candidate) / 100.0).moneyText)
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 90, alignment: .trailing)
                        }
                    }
                }
            }
            SectorExposureView(candidates: candidates, allocations: allocations)
        }
    }

    private var totalAllocation: Double {
        candidates.reduce(0) { $0 + allocation(for: $1) }
    }

    private var expectedUpside: Double {
        candidates.reduce(0) { total, candidate in
            total + accountSize * allocation(for: candidate) / 100.0 * max(candidate.upsidePercent, 0) / 100.0
        }
    }

    private var stopRisk: Double {
        candidates.reduce(0) { total, candidate in
            total + accountSize * allocation(for: candidate) / 100.0 * abs(min(candidate.stopDistancePercent, 0)) / 100.0
        }
    }

    private var maxLossWarning: String {
        stopRisk / max(accountSize, 1) > 0.08 ? "reduce size" : "within guardrail"
    }

    private func allocation(for candidate: Candidate) -> Double {
        allocations[candidate.ticker, default: candidate.positionSizePct ?? candidate.kellySizePct ?? 0]
    }
}

struct SectorExposureView: View {
    let candidates: [Candidate]
    let allocations: [String: Double]

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 10) {
                Text("Sector Concentration")
                    .font(.headline)
                ForEach(sectorRows, id: \.sector) { row in
                    HStack {
                        Text(row.sector)
                            .frame(width: 150, alignment: .leading)
                        MiniBar(value: row.weight / max(total, 1), tint: row.weight > 40 ? Design.red : Design.blue)
                        Text(row.weight.percentText)
                            .font(.caption.monospacedDigit())
                            .frame(width: 54, alignment: .trailing)
                    }
                }
            }
        }
    }

    private var total: Double {
        sectorRows.reduce(0) { $0 + $1.weight }
    }

    private var sectorRows: [(sector: String, weight: Double)] {
        let grouped = Dictionary(grouping: candidates, by: { $0.sector ?? "Unknown" })
        return grouped.map { sector, names in
            (sector, names.reduce(0) { $0 + allocations[$1.ticker, default: $1.positionSizePct ?? $1.kellySizePct ?? 0] })
        }
        .sorted { $0.weight > $1.weight }
    }
}

struct WhyNotPickedView: View {
    let candidates: [Candidate]
    let onOpen: (Candidate) -> Void

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Why Not Picked?")
                        .font(.headline)
                    Spacer()
                    Pill(text: "\(candidates.count) failed", color: Design.yellow)
                }
                if candidates.isEmpty {
                    EmptyState(text: "No failed high-profile candidates were available in the current scan artifact.")
                } else {
                    ForEach(candidates) { candidate in
                        VStack(alignment: .leading, spacing: 7) {
                            HStack {
                                Button(candidate.ticker) {
                                    onOpen(candidate)
                                }
                                .buttonStyle(.plain)
                                .font(.callout.weight(.semibold))
                                Text(candidate.companyName ?? candidate.sector ?? "")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Spacer()
                                if let marketCap = candidate.marketCap, marketCap > 0 {
                                    Text(marketCap.marketCapText)
                                        .font(.caption.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                }
                                Text(String(format: "%.1f", candidate.finalScore ?? 0))
                                    .font(.callout.monospacedDigit().weight(.semibold))
                            }
                            FlowLayout(items: rejectionReasons(for: candidate))
                            Divider().opacity(0.45)
                        }
                    }
                }
            }
        }
    }

    private func rejectionReasons(for candidate: Candidate) -> [String] {
        if let why = candidate.diagnostics?.analyst?.whyNotOfficial, !why.isEmpty {
            return [why] + computedReasons(for: candidate)
        }
        return computedReasons(for: candidate)
    }

    private func computedReasons(for candidate: Candidate) -> [String] {
        var reasons: [String] = []
        if (candidate.rsScore ?? 100) < 55 { reasons.append("Weak RS") }
        if (candidate.riskReward ?? 0) < 1.2 { reasons.append("Bad risk/reward") }
        if (candidate.volumeMomentumScore ?? 100) < 55 { reasons.append("Low volume") }
        if (candidate.sectorTailwindPoints ?? 0) < 0 { reasons.append("Sector drag") }
        if abs(candidate.stopDistancePercent) > 8 { reasons.append("Too much volatility") }
        if reasons.isEmpty { reasons.append("Below top-10 rank") }
        return reasons
    }
}

struct FlowLayout: View {
    let items: [String]

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 7) { pills }
            VStack(alignment: .leading, spacing: 7) { pills }
        }
    }

    @ViewBuilder
    private var pills: some View {
        ForEach(items.prefix(5), id: \.self) { item in
            Pill(text: item, color: Design.yellow)
        }
    }
}

struct PickCard: View {
    let candidate: Candidate
    var rank: Int

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text("#\(rank)")
                        .font(.caption.monospacedDigit().weight(.bold))
                        .foregroundStyle(scoreColor)
                        .frame(width: 32, alignment: .leading)
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text(candidate.ticker)
                                .font(.system(size: 24, weight: .semibold, design: .rounded))
                            Text(candidate.companyName ?? "Unknown company")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        HStack(spacing: 8) {
                            Pill(text: candidate.sector ?? "Unknown", color: Design.blue)
                            Pill(text: candidate.tierLabel ?? "Tier", color: Design.purple)
                            Pill(text: candidate.scoreBand, color: scoreColor)
                        }
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 6) {
                        Text(String(format: "%.1f", candidate.finalScore ?? 0))
                            .font(.title3.monospacedDigit().weight(.semibold))
                            .foregroundStyle(scoreColor)
                        Text("score")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .textCase(.uppercase)
                    }
                }

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 122), spacing: 10)], spacing: 10) {
                    RiskCell(title: "Current", value: (candidate.currentPrice ?? 0).moneyText)
                    RiskCell(title: "Stop", value: (candidate.stopLoss ?? 0).moneyText)
                    RiskCell(title: "Target", value: candidate.targetPrice.moneyText)
                    RiskCell(title: "Upside", value: candidate.upsidePercent.percentText)
                }

                ScoreFactorStrip(candidate: candidate)

                HStack(spacing: 8) {
                    MiniBar(value: min(max((candidate.finalScore ?? 0) / 100, 0), 1), tint: scoreColor)
                        .frame(maxWidth: 190)
                    Text("Confluence \(candidate.confluenceCount ?? 0)/5")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Stop \(candidate.stopDistancePercent.percentText)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text("R:R \(String(format: "%.1f", candidate.riskReward ?? 0))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(Design.teal)
                    Spacer()
                    Text(candidate.upsidePercent.percentText)
                        .font(.callout.monospacedDigit().weight(.semibold))
                        .foregroundStyle(candidate.upsidePercent >= 0 ? Design.green : Design.red)
                }

                if let explanation = candidate.aiExplanation, !explanation.isEmpty {
                    Text(explanation)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(4)
                }

                if let analyst = candidate.diagnostics?.analyst {
                    Divider()
                    VStack(alignment: .leading, spacing: 9) {
                        HStack {
                            Label("Why this pick?", systemImage: "sparkles")
                                .font(.headline)
                            Spacer()
                            Pill(
                                text: String(format: "%.0f%% confidence", analyst.backtestConfidence ?? 0),
                                color: confidenceColor(analyst.backtestConfidence ?? 0)
                            )
                        }
                        BulletList(items: analyst.whyThisPick ?? [])
                        if let riskMap = analyst.riskRewardMap {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 112), spacing: 10)], spacing: 10) {
                                RiskCell(title: "Entry", value: riskMap.entry?.moneyText ?? "--")
                                RiskCell(title: "Stop", value: riskMap.stop?.moneyText ?? "--")
                                RiskCell(title: "Target", value: riskMap.target?.moneyText ?? "--")
                                RiskCell(title: "R:R", value: String(format: "%.1f", riskMap.riskReward ?? candidate.riskReward ?? 0))
                            }
                        }
                        DisclosureGroup {
                            AnalystDetailBlock(title: "What would invalidate it?", items: analyst.invalidation ?? [])
                            AnalystDetailBlock(title: "Similar historical trades", items: analyst.similarHistoricalSetups ?? [])
                            AnalystDetailBlock(title: "Watch-outs", items: analyst.negativeDrivers ?? [])
                            if let whyNot = analyst.whyNotOfficial, !whyNot.isEmpty {
                                Text(whyNot)
                                    .font(.callout)
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        } label: {
                            Label("Analyst notes", systemImage: "doc.text.magnifyingglass")
                                .font(.callout.weight(.semibold))
                        }
                    }
                }
            }
        }
    }

    private var scoreColor: Color {
        let score = candidate.finalScore ?? 0
        if score >= 70 { return Design.green }
        if score >= 54 { return Design.yellow }
        return Design.red
    }

    private func confidenceColor(_ value: Double) -> Color {
        if value >= 65 { return Design.green }
        if value >= 50 { return Design.yellow }
        return Design.red
    }
}

struct ScoreFactorStrip: View {
    let candidate: Candidate

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 138), spacing: 8)], spacing: 8) {
            ForEach(candidate.scoreBreakdown) { factor in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(factor.name)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        Spacer()
                        Text(String(format: "%.0f", factor.value))
                            .font(.caption.monospacedDigit().weight(.semibold))
                    }
                    MiniBar(value: factor.value / 100.0, tint: color(for: factor.colorName))
                }
                .padding(8)
                .background(.secondary.opacity(0.055), in: RoundedRectangle(cornerRadius: Design.radius))
            }
        }
    }

    private func color(for name: String) -> Color {
        switch name {
        case "green": Design.green
        case "teal": Design.teal
        case "purple": Design.purple
        case "yellow": Design.yellow
        default: Design.blue
        }
    }
}

struct BulletList: View {
    let items: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(items.prefix(4), id: \.self) { item in
                Label(item, systemImage: "checkmark.circle")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
    }
}

struct AnalystDetailBlock: View {
    let title: String
    let items: [String]

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                BulletList(items: items)
            }
            .padding(.top, 8)
        }
    }
}

struct RiskCell: View {
    @Environment(\.colorScheme) private var colorScheme
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Text(value)
                .font(.callout.monospacedDigit().weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.78)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(colorScheme == .dark ? .white.opacity(0.045) : .white.opacity(0.62), in: RoundedRectangle(cornerRadius: Design.radius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: Design.radius, style: .continuous)
                .stroke(.secondary.opacity(0.12), lineWidth: 1)
        }
    }
}
