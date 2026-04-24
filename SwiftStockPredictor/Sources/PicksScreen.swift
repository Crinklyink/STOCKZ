import SwiftUI

struct PicksScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader("Picks", subtitle: "Ranked trade candidates with score, upside, risk/reward, and model notes.")
                StatusStrip(
                    title: appState.candidates.isEmpty ? "No active watchlist" : "\(appState.candidates.count) candidates ready",
                    detail: appState.latestScan?.macroSummary ?? "Run a scan to refresh the current market context.",
                    symbol: "scope",
                    color: appState.candidates.isEmpty ? .secondary : .blue
                )

                if appState.candidates.isEmpty {
                    GlassPanel {
                        EmptyState(text: "No current candidates. Run Scan from the sidebar to generate this week's list.")
                    }
                } else {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 340), spacing: 14)], spacing: 14) {
                        ForEach(appState.candidates) { candidate in
                            PickCard(candidate: candidate)
                        }
                    }
                }
            }
            .padding(.bottom, 24)
        }
    }
}

struct PickCard: View {
    let candidate: Candidate

    var body: some View {
        GlassPanel {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(candidate.ticker)
                            .font(.system(size: 32, weight: .semibold, design: .rounded))
                        Text(candidate.companyName ?? "Unknown company")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Pill(text: String(format: "Score %.1f", candidate.finalScore ?? 0), color: scoreColor)
                }

                HStack {
                    VStack(alignment: .leading) {
                        Text("Current")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text((candidate.currentPrice ?? 0).moneyText)
                            .font(.title3.monospacedDigit())
                    }
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("Target")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(candidate.targetPrice.moneyText)
                            .font(.title3.monospacedDigit())
                    }
                }

                ProgressView(value: min(max((candidate.upsidePercent + 10) / 30, 0), 1))
                    .tint(candidate.upsidePercent >= 0 ? Design.green : Design.red)
                HStack(spacing: 8) {
                    MiniBar(value: min(max((candidate.finalScore ?? 0) / 100, 0), 1), tint: scoreColor)
                    Text("Confluence \(candidate.confluenceCount ?? 0)/5")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                HStack {
                    Pill(text: candidate.sector ?? "Unknown", color: .blue)
                    Pill(text: candidate.tierLabel ?? "Tier", color: .purple)
                    Pill(text: "R:R \(String(format: "%.1f", candidate.riskReward ?? 0))", color: .teal)
                    Spacer()
                    Text(candidate.upsidePercent.percentText)
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(candidate.upsidePercent >= 0 ? Design.green : Design.red)
                }

                if let explanation = candidate.aiExplanation, !explanation.isEmpty {
                    Text(explanation)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(4)
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
}
