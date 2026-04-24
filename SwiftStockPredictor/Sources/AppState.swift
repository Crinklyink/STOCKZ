import Foundation
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    @Published var latestScan: LatestScan?
    @Published var modelMetadata: ModelMetadata?
    @Published var trainingEvents: [TrainingEvent] = []
    @Published var currentJob: String = "Idle"
    @Published var currentDetail: String = "Ready"
    @Published var progress: Double = 0
    @Published var isRunning = false
    @Published var autoTrainerEnabled = true
    @Published var selectedUniverse = "full"
    @Published var useFreshData = false
    @Published var lastError: String?

    let projectRoot: URL

    init() {
        let envRoot = ProcessInfo.processInfo.environment["STOCK_PREDICTOR_ROOT"]
        if let envRoot, !envRoot.isEmpty {
            projectRoot = URL(fileURLWithPath: envRoot)
        } else {
            projectRoot = Bundle.main.bundleURL
                .deletingLastPathComponent()
                .deletingLastPathComponent()
        }
    }

    var candidates: [Candidate] {
        latestScan?.candidates ?? []
    }

    var summary: ScanSummary? {
        latestScan?.scanSummary
    }

    func reload() async {
        latestScan = loadJSON("stock_predictor/artifacts/latest_scan.json", as: LatestScan.self)
        let adaptive = loadJSON("stock_predictor/models/adaptive_model_metadata.json", as: ModelMetadata.self)
        let xgb = loadJSON("stock_predictor/models/xgboost_metadata.json", as: ModelMetadata.self)
        modelMetadata = adaptive ?? xgb ?? latestScan?.trainingReport.map(ModelMetadata.init(report:))
        loadEnvSettings()
    }

    func runTraining() {
        runBackendJob(flag: "--train", title: "Training model")
    }

    func runBacktest() {
        runBackendJob(flag: "--backtest-adaptive", title: "Rolling backtest")
    }

    func runScan() {
        runBackendJob(flag: "--top-n", title: "Weekly scan", extra: ["10"])
    }

    func toggleAutoTrainer(_ enabled: Bool) {
        autoTrainerEnabled = enabled
        writeEnvValue(key: "RETRAIN_MODEL_WEEKLY", value: enabled ? "1" : "0")
    }

    func setDefaultUniverse(_ universe: String) {
        selectedUniverse = universe
        writeEnvValue(key: "DEFAULT_UNIVERSE", value: universe)
    }

    private func runBackendJob(flag: String, title: String, extra: [String] = []) {
        guard !isRunning else { return }
        isRunning = true
        progress = 0.04
        currentJob = title
        currentDetail = "Starting"
        trainingEvents.removeAll()

        let root = projectRoot
        let universe = selectedUniverse
        let fresh = useFreshData
        let pythonPath = backendPythonExecutable(projectRoot: root)

        Task.detached(priority: .userInitiated) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: pythonPath)
            var arguments = ["main.py", flag] + extra + ["--universe", universe]
            if fresh { arguments.append("--fresh") }
            process.arguments = arguments
            process.currentDirectoryURL = root
            process.environment = ProcessInfo.processInfo.environment.merging(
                ["PYTHONUNBUFFERED": "1", "STOCK_PREDICTOR_ROOT": root.path],
                uniquingKeysWith: { _, new in new }
            )

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            do {
                try process.run()
                await self.consume(text: "Using Python: \(pythonPath)")
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.finishJob(status: "Failed", detail: "Could not start: \(error.localizedDescription)")
                }
                return
            }

            let handle = pipe.fileHandleForReading
            var outputLines: [String] = []
            while process.isRunning {
                let data = handle.availableData
                if data.isEmpty { break }
                outputLines.append(contentsOf: Self.lines(from: data))
                await self.consume(data: data)
            }
            let remaining = handle.readDataToEndOfFile()
            if !remaining.isEmpty {
                await self.consume(data: remaining)
                outputLines.append(contentsOf: Self.lines(from: remaining))
            }
            process.waitUntilExit()

            await MainActor.run {
                if process.terminationStatus == 0 {
                    self.lastError = nil
                    self.finishJob(status: "Complete", detail: "Artifacts refreshed")
                    Task { await self.reload() }
                } else {
                    let tail = outputLines.suffix(5).joined(separator: " ")
                    let detail = self.backendFailureMessage(exitCode: process.terminationStatus, tail: tail)
                    self.lastError = detail
                    self.finishJob(status: "Failed", detail: detail)
                }
            }
        }
    }

    private func backendPythonExecutable(projectRoot: URL) -> String {
        let envPython = ProcessInfo.processInfo.environment["STOCK_PREDICTOR_PYTHON"]
        let candidates = [
            envPython,
            projectRoot.appendingPathComponent(".venv/bin/python3").path,
            projectRoot.appendingPathComponent(".venv/bin/python").path,
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3"
        ].compactMap { $0 }.filter { !$0.isEmpty }
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) } ?? "/usr/bin/python3"
    }

    private nonisolated static func lines(from data: Data) -> [String] {
        guard let chunk = String(data: data, encoding: .utf8) else { return [] }
        return chunk
            .split(whereSeparator: \.isNewline)
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    private func backendFailureMessage(exitCode: Int32, tail: String) -> String {
        if tail.contains("ModuleNotFoundError") || tail.contains("No module named") {
            return "Python dependencies are missing. Create .venv and install requirements.txt, or set STOCK_PREDICTOR_PYTHON."
        }
        if tail.isEmpty {
            return "Backend exited with code \(exitCode). No output was returned."
        }
        return "Backend exited with code \(exitCode): \(tail)"
    }

    private func consume(data: Data) async {
        let lines = Self.lines(from: data)
        await MainActor.run {
            for line in lines {
                trainingEvents.append(TrainingEvent(text: line))
                updateProgress(from: line)
            }
        }
    }

    private func consume(text: String) async {
        await MainActor.run {
            trainingEvents.append(TrainingEvent(text: text))
        }
    }

    private func updateProgress(from line: String) {
        let lower = line.lowercased()
        if lower.contains("resolving") { progress = max(progress, 0.12) }
        if lower.contains("fetching historical") { progress = max(progress, 0.26) }
        if lower.contains("sector") || lower.contains("macro") { progress = max(progress, 0.36) }
        if lower.contains("benchmark") { progress = max(progress, 0.44) }
        if lower.contains("calculating") { progress = max(progress, 0.54) }
        if lower.contains("training adaptive") { progress = max(progress, 0.70) }
        if lower.contains("walk-forward") || lower.contains("rolling backtest") { progress = max(progress, 0.82) }
        if lower.contains("saving") || lower.contains("writing") { progress = max(progress, 0.93) }
        if lower.contains("complete") { progress = 1.0 }
        if line.hasPrefix("[training]") || line.hasPrefix("[backtest]") {
            currentDetail = line
                .replacingOccurrences(of: "[training] ", with: "")
                .replacingOccurrences(of: "[backtest] ", with: "")
        }
    }

    private func finishJob(status: String, detail: String) {
        currentJob = status
        currentDetail = detail
        progress = status == "Complete" ? 1.0 : progress
        isRunning = false
    }

    private func loadJSON<T: Decodable>(_ relativePath: String, as type: T.Type) -> T? {
        let url = projectRoot.appendingPathComponent(relativePath)
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            lastError = "Could not read \(relativePath): \(error.localizedDescription)"
            return nil
        }
    }

    private func loadEnvSettings() {
        let url = projectRoot.appendingPathComponent(".env")
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return }
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            if parts[0] == "RETRAIN_MODEL_WEEKLY" {
                autoTrainerEnabled = parts[1] != "0"
            }
            if parts[0] == "DEFAULT_UNIVERSE" {
                selectedUniverse = parts[1]
            }
        }
    }

    private func writeEnvValue(key: String, value: String) {
        let url = projectRoot.appendingPathComponent(".env")
        var lines = (try? String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init)) ?? []
        var found = false
        for index in lines.indices {
            if lines[index].hasPrefix("\(key)=") {
                lines[index] = "\(key)=\(value)"
                found = true
            }
        }
        if !found { lines.append("\(key)=\(value)") }
        try? (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}

extension ModelMetadata {
    init(report: TrainingReport) {
        trained = report.trained
        trainedAt = report.trainedAt
        trainingSamples = report.trainingSamples
        validationSamples = report.validationSamples
        modelFamily = report.modelFamily
        selectedProfile = report.selectedProfile
        auc = report.auc
        ensembleAuc = report.ensembleAuc
        xgbAuc = report.xgbAuc
        lightgbmAuc = report.lightgbmAuc
        ensembleWeights = nil
        featureImportance = report.featureImportance
    }
}
