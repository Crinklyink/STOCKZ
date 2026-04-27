import SwiftUI

enum TrainingMode {
    case training
    case backtesting
}

struct TrainingScreen: View {
    @EnvironmentObject private var appState: AppState
    let mode: TrainingMode

    var body: some View {
        PageFrame {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader(
                    mode == .training ? "Training" : "Backtesting",
                    subtitle: mode == .training ? "Stable model runs with visible progress." : "Rolling validation and saved performance reports."
                )
                StatusStrip(
                    title: mode == .training ? "Model lab" : "Rolling validation",
                    detail: mode == .training
                        ? "Stable fallback mode is active: Random Forest, strict +6% target, native boosters disabled."
                        : "Validation reports are separate from the active fallback model card.",
                    symbol: mode == .training ? "cpu" : "chart.line.uptrend.xyaxis",
                    color: mode == .training ? Design.blue : Design.purple
                )

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 12)], spacing: 12) {
                    MetricTile(title: "Stack", value: appState.modelMetadata?.stackName ?? "--", footnote: appState.modelMetadata?.selectedProfile?.cleanedProfile ?? "profile")
                    MetricTile(title: "AUC", value: String(format: "%.3f", appState.modelMetadata?.activeAuc ?? 0), footnote: "walk-forward", tone: Design.green)
                    MetricTile(
                        title: "Target",
                        value: "+6%",
                        footnote: appState.modelMetadata?.targetSummary ?? "weekly barrier",
                        tone: .mint
                    )
                    MetricTile(title: "Samples", value: (appState.modelMetadata?.trainingSamples ?? 0).compactText, footnote: "training rows")
                    MetricTile(
                        title: "Status",
                        value: appState.isRunning ? "\(appState.elapsedSeconds)s" : appState.currentJob,
                        footnote: appState.currentDetail,
                        tone: appState.isRunning ? Design.blue : .primary
                    )
                }

                RuntimeModePanel()

                GlassPanel {
                    VStack(alignment: .leading, spacing: 16) {
                        HStack(alignment: .center, spacing: 14) {
                            Picker("Universe", selection: Binding(
                                get: { appState.selectedUniverse },
                                set: { appState.setDefaultUniverse($0) }
                            )) {
                                Text("Mini").tag("mini")
                                Text("Full").tag("full")
                                Text("US Market").tag("us_market")
                            }
                            .pickerStyle(.segmented)
                            .frame(minWidth: 260, idealWidth: 330, maxWidth: 360)

                            Toggle("Fresh data", isOn: $appState.useFreshData)
                            Toggle("Auto-train weekly", isOn: Binding(
                                get: { appState.autoTrainerEnabled },
                                set: { appState.toggleAutoTrainer($0) }
                            ))

                            Spacer()

                            if appState.lastError?.contains("Python dependencies") == true {
                                Button {
                                    appState.setupPythonEnvironment()
                                } label: {
                                    Label("Set Up Python", systemImage: "wrench.and.screwdriver")
                                }
                                .buttonStyle(.bordered)
                                .disabled(appState.isRunning)
                            }

                            Button {
                                mode == .training ? appState.runTraining() : appState.runBacktest()
                            } label: {
                                Label(mode == .training ? "Train Model" : "Run Backtest", systemImage: "play.fill")
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(appState.isRunning)
                        }
                        .lineLimit(1)
                        .minimumScaleFactor(0.78)

                        ProgressView(value: appState.progress)
                            .tint(.blue)
                        HStack {
                            Label(appState.currentDetail, systemImage: appState.isRunning ? "arrow.triangle.2.circlepath" : "checkmark.circle")
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text("\(Int(appState.progress * 100))%")
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                GlassPanel {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            SectionHeader("Live Scan Progress")
                            Spacer()
                            Pill(text: appState.isRunning ? "Running" : "Idle", color: appState.isRunning ? .blue : .secondary)
                        }
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 12)], spacing: 12) {
                            ForEach(appState.scanStages) { stage in
                                ScanStageCard(stage: stage)
                            }
                        }
                        if let last = appState.trainingEvents.last?.text {
                            Divider().opacity(0.35)
                            Label(last, systemImage: "waveform.path.ecg")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    }
                }
            }
        }
    }
}

struct ScanStageCard: View {
    let stage: ScanStage

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: stage.symbol)
                    .foregroundStyle(tone)
                    .frame(width: 28, height: 28)
                    .background(tone.opacity(0.14), in: RoundedRectangle(cornerRadius: Design.radius))
                Spacer()
                Image(systemName: statusSymbol)
                    .foregroundStyle(tone)
            }
            Text(stage.title)
                .font(.callout.weight(.semibold))
            Text(stage.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, minHeight: 112, alignment: .topLeading)
        .padding(14)
        .background(tone.opacity(stage.status == .pending ? 0.035 : 0.075), in: RoundedRectangle(cornerRadius: Design.radius))
        .overlay {
            RoundedRectangle(cornerRadius: Design.radius)
                .stroke(tone.opacity(stage.status == .pending ? 0.12 : 0.28), lineWidth: 1)
        }
    }

    private var tone: Color {
        switch stage.status {
        case .pending: .secondary
        case .active: Design.blue
        case .complete: Design.green
        }
    }

    private var statusSymbol: String {
        switch stage.status {
        case .pending: "circle"
        case .active: "arrow.triangle.2.circlepath"
        case .complete: "checkmark.circle.fill"
        }
    }
}

struct RuntimeModePanel: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        GlassPanel {
            HStack(alignment: .top, spacing: 18) {
                VStack(alignment: .leading, spacing: 10) {
                    SectionHeader("Runtime Mode")
                    InfoRow(title: "Active model", value: appState.activeModelSummary, symbol: "cpu", tone: Design.blue)
                    InfoRow(title: "Native boosters", value: "Disabled for stability", symbol: "shield.lefthalf.filled", tone: Design.green)
                    InfoRow(title: "FinBERT sentiment", value: "Off unless explicitly enabled", symbol: "text.bubble", tone: Design.teal)
                }
                Divider()
                VStack(alignment: .leading, spacing: 12) {
                    SectionHeader("Backend Contract")
                    StepPill(number: 1, title: "Safe fit", detail: "Avoids local OpenMP crashes")
                    StepPill(number: 2, title: "Strict target", detail: "+6% before -3.5% stop", color: Design.green)
                    StepPill(number: 3, title: "Refresh scan", detail: "Updates picks after training", color: Design.teal)
                }
                .frame(width: 320)
            }
        }
    }
}
