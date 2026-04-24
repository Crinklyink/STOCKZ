import SwiftUI

struct DashboardScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader("Dashboard", subtitle: "A clean read on picks, regime, model quality, and recent training state.")
                StatusStrip(
                    title: dashboardStatusTitle,
                    detail: dashboardStatusDetail,
                    symbol: appState.lastError == nil ? "checkmark.seal" : "exclamationmark.triangle",
                    color: appState.lastError == nil ? .blue : Design.red
                )

                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 4), spacing: 14) {
                    MetricTile(
                        title: "This Week",
                        value: "\(appState.candidates.count)",
                        footnote: appState.candidates.isEmpty ? "No current picks" : "ranked names",
                        tone: appState.candidates.isEmpty ? Design.secondary : Design.green
                    )
                    MetricTile(
                        title: "Model AUC",
                        value: String(format: "%.3f", appState.modelMetadata?.activeAuc ?? 0),
                        footnote: appState.modelMetadata?.selectedProfile?.cleanedProfile ?? "training profile",
                        tone: (appState.modelMetadata?.activeAuc ?? 0) >= 0.6 ? Design.green : Design.yellow
                    )
                    MetricTile(
                        title: "VIX",
                        value: String(format: "%.1f", appState.summary?.vix ?? 0),
                        footnote: appState.summary?.regime ?? "market regime",
                        tone: (appState.summary?.vix ?? 0) > 25 ? Design.red : Design.green
                    )
                    MetricTile(
                        title: "Samples",
                        value: "\(appState.modelMetadata?.trainingSamples ?? 0)",
                        footnote: appState.modelMetadata?.trainedAt?.prefix(10).description ?? "training rows",
                        tone: .primary
                    )
                }

                HStack(alignment: .top, spacing: 14) {
                    GlassPanel {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                SectionHeader("Watchlist")
                                Spacer()
                                Pill(text: "\(appState.candidates.count) names")
                            }
                            if appState.candidates.isEmpty {
                                EmptyState(text: "No picks have been generated yet. Run a scan or train the model first.")
                            } else {
                                ForEach(appState.candidates.prefix(6)) { candidate in
                                    CandidateRow(candidate: candidate)
                                    Divider()
                                }
                            }
                        }
                    }

                    GlassPanel {
                        VStack(alignment: .leading, spacing: 18) {
                            SectionHeader("Market Regime")
                            RegimeGauge(value: appState.summary?.vix ?? 0)
                            HStack {
                                Label("SPY", systemImage: "chart.line.uptrend.xyaxis")
                                Spacer()
                                Text((appState.summary?.spyWeekReturn ?? 0).percentText)
                            }
                            HStack {
                                Label("Regime", systemImage: "waveform.path.ecg")
                                Spacer()
                                Text(appState.latestScan?.regimeLabel ?? appState.summary?.regime ?? "Neutral")
                            }
                        }
                    }
                    .frame(width: 380)
                }

                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 3), spacing: 14) {
                    MetricTile(
                        title: "Target Hit Rate",
                        value: String(format: "%.0f%%", appState.latestScan?.paperTradeSummary?.targetHitRate ?? 0),
                        footnote: "\(appState.latestScan?.paperTradeSummary?.weeks ?? 0) rolling weeks",
                        tone: (appState.latestScan?.paperTradeSummary?.targetHitRate ?? 0) >= 55 ? Design.green : Design.yellow
                    )
                    MetricTile(
                        title: "Avg Return",
                        value: (appState.latestScan?.paperTradeSummary?.averageReturn ?? 0).percentText,
                        footnote: "paper-trade tracking",
                        tone: (appState.latestScan?.paperTradeSummary?.averageReturn ?? 0) >= 0 ? Design.green : Design.red
                    )
                    MetricTile(
                        title: "Best Pick",
                        value: appState.latestScan?.paperTradeSummary?.bestPick ?? "--",
                        footnote: (appState.latestScan?.paperTradeSummary?.bestReturn ?? 0).percentText,
                        tone: Design.green
                    )
                }
            }
            .padding(.bottom, 24)
        }
    }

    private var dashboardStatusTitle: String {
        if let error = appState.lastError, !error.isEmpty { return "Needs attention" }
        if appState.candidates.isEmpty { return "Ready for a new scan" }
        return "\(appState.candidates.count) candidates loaded"
    }

    private var dashboardStatusDetail: String {
        if let error = appState.lastError, !error.isEmpty { return error }
        if let generated = appState.latestScan?.generatedAt {
            return "Latest artifact generated \(String(generated.prefix(16)).replacingOccurrences(of: "T", with: " "))"
        }
        return "No latest scan artifact yet. Use Run Scan or train the model first."
    }
}

struct CandidateRow: View {
    let candidate: Candidate

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(candidate.ticker)
                    .font(.headline)
                Text(candidate.companyName ?? candidate.sector ?? "Unknown")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Pill(text: String(format: "%.1f", candidate.finalScore ?? 0), color: scoreColor)
            VStack(alignment: .trailing, spacing: 3) {
                Text((candidate.currentPrice ?? 0).moneyText)
                Text(candidate.upsidePercent.percentText)
                    .foregroundStyle(candidate.upsidePercent >= 0 ? Design.green : Design.red)
            }
            .font(.callout.monospacedDigit())
        }
        .padding(.vertical, 4)
    }

    private var scoreColor: Color {
        let score = candidate.finalScore ?? 0
        if score >= 70 { return Design.green }
        if score >= 54 { return Design.yellow }
        return Design.red
    }
}

struct RegimeGauge: View {
    @Environment(\.colorScheme) private var colorScheme
    let value: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("VIX")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1f", value))
                    .font(.system(size: 46, weight: .light, design: .rounded))
                    .monospacedDigit()
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(LinearGradient(colors: [Design.green, Design.yellow, Design.red], startPoint: .leading, endPoint: .trailing))
                    Circle()
                        .fill(colorScheme == .dark ? Color.white : Color.white)
                        .shadow(radius: 4)
                        .frame(width: 16, height: 16)
                        .offset(x: max(0, min(proxy.size.width - 16, proxy.size.width * value / 40 - 8)))
                }
            }
            .frame(height: 12)
            HStack {
                Text("Calm")
                Spacer()
                Text("Caution")
                Spacer()
                Text("Risk")
            }
            .font(.caption2.weight(.medium))
            .foregroundStyle(.secondary)
        }
    }
}

struct EmptyState: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, minHeight: 160)
    }
}
