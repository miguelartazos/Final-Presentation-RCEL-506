import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { BEBotDifficulty } from '../src/types/be-game.types';
import type { BEBotLinearModel } from '../src/utils/beBotModel';
import type { BEBotPolicyConfig, BEBotPolicyRuntimeOptions } from '../src/utils/beBotPolicy';
import { HARD_TUNED_BOT_POLICY_CONFIG } from '../src/utils/beBotPolicy';
import { BOT_MODEL_FEATURES, EXPERT_BOT_LINEAR_MODEL } from '../src/utils/beBotModel.generated';
import { runPolicyBenchmark, summarize } from './be-bot-selfplay';

type OptimizerRow = {
  candidateId: string;
  split: 'train' | 'validation' | 'final';
  games: number;
  averageScore: number;
  baselineAverageScore: number;
  scoreLift: number;
  averageRank: number;
  baselineAverageRank: number;
  rankLift: number;
  winRate: number;
  baselineWinRate: number;
  winRateLift: number;
  invalidActionCount: number;
  noOpCount: number;
  safeNoOpRate: number;
  financeExploitCount: number;
  averageGameLength: number;
  objective: number;
};

type EvaluationResult = {
  row: OptimizerRow;
  runtimeOptions?: BEBotPolicyRuntimeOptions;
};

const ARTIFACT_DIR = join(process.cwd(), 'training', 'bot-ml', 'artifacts');

function outputDirForMode(mode: 'expert' | 'hard'): string {
  return join(
    process.cwd(),
    'training',
    'bot-ml',
    'experiments',
    mode === 'hard' ? 'hard-optimizer-latest' : 'optimizer-latest',
  );
}

function ensureOutputDirs(mode: 'expert' | 'hard'): void {
  mkdirSync(outputDirForMode(mode), { recursive: true });
  mkdirSync(ARTIFACT_DIR, { recursive: true });
}

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

function getArgNumber(flag: string, fallback: number): number {
  const index = process.argv.findIndex((arg) => arg === flag);
  if (index < 0) return fallback;
  const parsed = Number(process.argv[index + 1]);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function getMode(): 'expert' | 'hard' {
  const modeIndex = process.argv.findIndex((arg) => arg === '--mode' || arg === '--policy');
  return process.argv[modeIndex + 1] === 'hard' ? 'hard' : 'expert';
}

function buildSeeds(count: number, offset = 0): number[] {
  return Array.from({ length: count }, (_, index) => 101 + (offset + index) * 101);
}

function csvEscape(value: unknown): string {
  const text = String(value ?? '');
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv<T extends Record<string, unknown>>(rows: T[], columns: Array<keyof T>): string {
  return [
    columns.map(String).join(','),
    ...rows.map((row) => columns.map((column) => csvEscape(row[column])).join(',')),
  ].join('\n');
}

function cloneBaseModel(): BEBotLinearModel {
  const source = EXPERT_BOT_LINEAR_MODEL as BEBotLinearModel;
  return {
    intercept: source.intercept,
    coefficients: { ...source.coefficients },
    means: { ...source.means },
    scales: { ...source.scales },
  };
}

function cloneHardConfig(): BEBotPolicyConfig {
  return {
    version: HARD_TUNED_BOT_POLICY_CONFIG.version,
    weights: { ...HARD_TUNED_BOT_POLICY_CONFIG.weights },
    planner: { ...HARD_TUNED_BOT_POLICY_CONFIG.planner },
  };
}

function sampleBetween(random: () => number, min: number, max: number): number {
  return min + random() * (max - min);
}

function makeCandidateModel(candidateIndex: number, random: () => number): BEBotLinearModel {
  const model = cloneBaseModel();
  if (candidateIndex === 0) return model;

  model.coefficients = {
    cash: sampleBetween(random, -0.05, 0.18),
    brand: sampleBetween(random, 0, 0.25),
    netProfit: sampleBetween(random, 0, 0.2),
    businessCount: sampleBetween(random, -0.1, 0.55),
    handCount: sampleBetween(random, -0.1, 0.18),
    actionCost: sampleBetween(random, -1.5, -0.15),
    actionIncome: sampleBetween(random, 0.05, 0.9),
    actionVP: sampleBetween(random, 0.1, 1.1),
    placementAffinity: sampleBetween(random, 0, 0.7),
    usesTemporaryFocus: sampleBetween(random, -0.1, 0.25),
  };
  return model;
}

function makeCandidateHardConfig(candidateIndex: number, random: () => number): BEBotPolicyConfig {
  const config = cloneHardConfig();
  if (candidateIndex === 0) return config;

  config.version = `hard-autoresearch-tuned-candidate-${candidateIndex}`;
  config.weights = {
    cost: sampleBetween(random, -0.0024, -0.00005),
    income: sampleBetween(random, 0.00005, 0.0011),
    valuationPoints: sampleBetween(random, 0.05, 1.2),
    placementAffinity: sampleBetween(random, 0, 0.8),
    brand: sampleBetween(random, -0.05, 0.3),
    handPressure: sampleBetween(random, -0.1, 0.3),
    temporaryFocus: sampleBetween(random, -0.05, 0.3),
    projectedNetValue: sampleBetween(random, -0.00005, 0.00065),
    remainingQuarters: sampleBetween(random, -0.15, 0.45),
    cashAfterAction: sampleBetween(random, -0.00004, 0.00012),
    staffFit: sampleBetween(random, -0.2, 0.8),
    launchAffordability: sampleBetween(random, -0.2, 1),
    departmentHasLegalAction: sampleBetween(random, 0, 2.2),
    cashNeed: sampleBetween(random, 0, 0.0007),
    loanCapacity: sampleBetween(random, -0.2, 0.5),
    debtPenaltyProjected: sampleBetween(random, -0.8, 0),
    runwayAfterAction: sampleBetween(random, -0.00005, 0.00018),
    financeExploitRisk: sampleBetween(random, -12, -2),
  };
  config.planner = {
    oportunidadesNeed: sampleBetween(random, 0.1, 3.5),
    lanzarLegalAction: sampleBetween(random, -1, 4),
    talentoUnderstaffed: sampleBetween(random, 0, 7),
    crecerPortfolio: sampleBetween(random, -0.2, 2.2),
    finanzasCashNeed: sampleBetween(random, 0, 8),
    finanzasLoanNeed: sampleBetween(random, 0, 5),
    noLegalActionPenalty: sampleBetween(random, -8, -0.2),
    healthyFinancePenalty: sampleBetween(random, -12, -1),
    safeNoOpPenalty: sampleBetween(random, -10, -1),
  };
  return config;
}

function evaluatePolicy(
  policy: BEBotDifficulty,
  seeds: number[],
  runtimeOptions?: BEBotPolicyRuntimeOptions,
) {
  const gameResults = runPolicyBenchmark(policy, seeds, runtimeOptions ? { runtimeOptions } : undefined);
  const rows = gameResults.rows;
  const noOpCount = rows.reduce((sum, row) => sum + row.safeNoOpCount, 0);
  return {
    rows,
    noOpCount,
    summary: summarize(rows).find((row) => row.policy === policy)!,
  };
}

function evaluateCandidate(params: {
  candidateId: string;
  split: OptimizerRow['split'];
  policy: BEBotDifficulty;
  seeds: number[];
  baseline: ReturnType<typeof evaluatePolicy>;
  runtimeOptions?: BEBotPolicyRuntimeOptions;
}): EvaluationResult {
  const candidate = evaluatePolicy(params.policy, params.seeds, params.runtimeOptions);
  const scoreLift = candidate.summary.averageScore - params.baseline.summary.averageScore;
  const winRateLift = candidate.summary.winRate - params.baseline.summary.winRate;
  const rankLift = params.baseline.summary.averageRank - candidate.summary.averageRank;
  const objective =
    scoreLift +
    rankLift * 0.75 +
    winRateLift * 2 -
    candidate.summary.invalidActionCount * 100 -
    candidate.summary.safeNoOpRate * 8 -
    candidate.summary.financeExploitCount * 0.05 -
    Math.max(0, candidate.summary.averageGameLength - 80) * 0.1;

  return {
    runtimeOptions: params.runtimeOptions,
    row: {
      candidateId: params.candidateId,
      split: params.split,
      games: candidate.summary.games,
      averageScore: candidate.summary.averageScore,
      baselineAverageScore: params.baseline.summary.averageScore,
      scoreLift,
      averageRank: candidate.summary.averageRank,
      baselineAverageRank: params.baseline.summary.averageRank,
      rankLift,
      winRate: candidate.summary.winRate,
      baselineWinRate: params.baseline.summary.winRate,
      winRateLift,
      invalidActionCount: candidate.summary.invalidActionCount,
      noOpCount: candidate.noOpCount,
      safeNoOpRate: candidate.summary.safeNoOpRate,
      financeExploitCount: candidate.summary.financeExploitCount,
      averageGameLength: candidate.summary.averageGameLength,
      objective,
    },
  };
}

function resultPassesHardGate(row: OptimizerRow): boolean {
  return row.scoreLift >= 0.25 &&
    row.winRateLift >= 0 &&
    row.invalidActionCount === 0;
}

function writeRows(mode: 'expert' | 'hard', rows: OptimizerRow[]): void {
  writeFileSync(
    join(outputDirForMode(mode), 'optimizer_results.csv'),
    `${toCsv(rows, [
      'candidateId',
      'split',
      'games',
      'averageScore',
      'baselineAverageScore',
      'scoreLift',
      'averageRank',
      'baselineAverageRank',
      'rankLift',
      'winRate',
      'baselineWinRate',
      'winRateLift',
      'invalidActionCount',
      'noOpCount',
      'safeNoOpRate',
      'financeExploitCount',
      'averageGameLength',
      'objective',
    ])}\n`,
  );
}

function optimizeExpert(): void {
  ensureOutputDirs('expert');
  const seedCount = getArgNumber('--seeds', 40);
  const candidateCount = getArgNumber('--candidates', 48);
  const random = seededRandom(getArgNumber('--optimizer-seed', 5001));
  const seeds = buildSeeds(seedCount);
  const baseline = evaluatePolicy('medium', seeds);
  const rows: OptimizerRow[] = [];
  let bestModel = cloneBaseModel();
  let bestRow: OptimizerRow | null = null;

  for (let i = 0; i < candidateCount; i++) {
    const model = makeCandidateModel(i, random);
    const result = evaluateCandidate({
      candidateId: i === 0 ? 'current-generated-model' : `random-search-${i}`,
      split: 'train',
      policy: 'expert',
      seeds,
      baseline,
      runtimeOptions: { expertModel: model },
    });
    rows.push(result.row);
    if (!bestRow || result.row.objective > bestRow.objective) {
      bestRow = result.row;
      bestModel = model;
    }
  }

  rows.sort((a, b) => b.objective - a.objective);
  const summary = [
    '# Expert Bot Coefficient Optimization',
    '',
    `Seeds: ${seedCount}`,
    `Candidates: ${candidateCount}`,
    `Best candidate: ${bestRow?.candidateId ?? 'none'}`,
    `Best score lift: ${(bestRow?.scoreLift ?? 0).toFixed(3)}`,
    `Best rank lift: ${(bestRow?.rankLift ?? 0).toFixed(3)}`,
    `Best win-rate lift: ${(((bestRow?.winRateLift ?? 0) * 100)).toFixed(1)}%`,
    `Invalid actions: ${bestRow?.invalidActionCount ?? 0}`,
    `Safe no-op rate: ${(((bestRow?.safeNoOpRate ?? 0) * 100)).toFixed(1)}%`,
    '',
    '## Best Coefficients',
    '',
    ...BOT_MODEL_FEATURES.map((feature) => `- ${feature}: ${bestModel.coefficients[feature].toFixed(4)}`),
    '',
    '## Interpretation',
    '',
    'The optimizer searches explainable linear model coefficients. It does not change rules, cards, or UI.',
  ].join('\n');

  writeRows('expert', rows);
  writeFileSync(join(outputDirForMode('expert'), 'best_expert_model.json'), `${JSON.stringify(bestModel, null, 2)}\n`);
  writeFileSync(join(outputDirForMode('expert'), 'summary.md'), `${summary}\n`);
  writeFileSync(join(ARTIFACT_DIR, 'optimized-expert-model.json'), `${JSON.stringify(bestModel, null, 2)}\n`);

  console.log('Business Empire expert bot optimization complete');
  console.log(
    `baseline avg=${baseline.summary.averageScore.toFixed(2)}, best avg=${(bestRow?.averageScore ?? 0).toFixed(2)}, scoreLift=${(bestRow?.scoreLift ?? 0).toFixed(2)}, rankLift=${(bestRow?.rankLift ?? 0).toFixed(2)}, winRateLift=${(((bestRow?.winRateLift ?? 0) * 100)).toFixed(1)}%, invalid=${bestRow?.invalidActionCount ?? 0}`,
  );
  console.log(`Artifacts: ${outputDirForMode('expert')}`);
}

function optimizeHard(): void {
  ensureOutputDirs('hard');
  const trainCount = getArgNumber('--train-seeds', 40);
  const validationCount = getArgNumber('--validation-seeds', 60);
  const finalCount = getArgNumber('--final-seeds', 100);
  const candidateCount = getArgNumber('--candidates', 96);
  const random = seededRandom(getArgNumber('--optimizer-seed', 7001));
  const trainSeeds = buildSeeds(trainCount, 0);
  const validationSeeds = buildSeeds(validationCount, trainCount);
  const finalSeeds = buildSeeds(finalCount, trainCount + validationCount);
  const trainBaseline = evaluatePolicy('medium', trainSeeds);
  const validationBaseline = evaluatePolicy('medium', validationSeeds);
  const finalBaseline = evaluatePolicy('medium', finalSeeds);
  const trainRows: OptimizerRow[] = [];
  let bestTrain: EvaluationResult | null = null;
  let bestConfig = cloneHardConfig();

  for (let i = 0; i < candidateCount; i++) {
    const policyConfig = makeCandidateHardConfig(i, random);
    const result = evaluateCandidate({
      candidateId: i === 0 ? 'current-hard-policy' : `hard-random-search-${i}`,
      split: 'train',
      policy: 'hard',
      seeds: trainSeeds,
      baseline: trainBaseline,
      runtimeOptions: { policyConfig },
    });
    trainRows.push(result.row);
    if (!bestTrain || result.row.objective > bestTrain.row.objective) {
      bestTrain = result;
      bestConfig = policyConfig;
    }
  }

  const validation = evaluateCandidate({
    candidateId: bestTrain?.row.candidateId ?? 'none',
    split: 'validation',
    policy: 'hard',
    seeds: validationSeeds,
    baseline: validationBaseline,
    runtimeOptions: { policyConfig: bestConfig },
  });
  const final = evaluateCandidate({
    candidateId: bestTrain?.row.candidateId ?? 'none',
    split: 'final',
    policy: 'hard',
    seeds: finalSeeds,
    baseline: finalBaseline,
    runtimeOptions: { policyConfig: bestConfig },
  });
  const allRows = [...trainRows.sort((a, b) => b.objective - a.objective), validation.row, final.row];
  const accepted = resultPassesHardGate(validation.row) && resultPassesHardGate(final.row);
  const summary = [
    '# Hard Bot Heuristic Optimization',
    '',
    `Train seeds: ${trainCount}`,
    `Validation seeds: ${validationCount}`,
    `Final test seeds: ${finalCount}`,
    `Candidates: ${candidateCount}`,
    `Best train candidate: ${bestTrain?.row.candidateId ?? 'none'}`,
    `Validation and final accepted: ${accepted ? 'yes' : 'no'}`,
    '',
    '## Train Result',
    '',
    `- Score lift: ${(bestTrain?.row.scoreLift ?? 0).toFixed(3)}`,
    `- Rank lift: ${(bestTrain?.row.rankLift ?? 0).toFixed(3)}`,
    `- Win-rate lift: ${(((bestTrain?.row.winRateLift ?? 0) * 100)).toFixed(1)}%`,
    `- Invalid actions: ${bestTrain?.row.invalidActionCount ?? 0}`,
    `- Safe no-op rate: ${(((bestTrain?.row.safeNoOpRate ?? 0) * 100)).toFixed(1)}%`,
    '',
    '## Validation Result',
    '',
    `- Medium baseline average score: ${validation.row.baselineAverageScore.toFixed(3)}`,
    `- Hard average score: ${validation.row.averageScore.toFixed(3)}`,
    `- Score lift: ${validation.row.scoreLift.toFixed(3)}`,
    `- Rank lift: ${validation.row.rankLift.toFixed(3)}`,
    `- Win-rate lift: ${(validation.row.winRateLift * 100).toFixed(1)}%`,
    `- Invalid actions: ${validation.row.invalidActionCount}`,
    `- Safe no-ops: ${validation.row.noOpCount}`,
    `- Safe no-op rate: ${(validation.row.safeNoOpRate * 100).toFixed(1)}%`,
    `- Finance exploit count: ${validation.row.financeExploitCount}`,
    '',
    '## Final Test Result',
    '',
    `- Medium baseline average score: ${final.row.baselineAverageScore.toFixed(3)}`,
    `- Hard average score: ${final.row.averageScore.toFixed(3)}`,
    `- Score lift: ${final.row.scoreLift.toFixed(3)}`,
    `- Rank lift: ${final.row.rankLift.toFixed(3)}`,
    `- Win-rate lift: ${(final.row.winRateLift * 100).toFixed(1)}%`,
    `- Invalid actions: ${final.row.invalidActionCount}`,
    `- Safe no-ops: ${final.row.noOpCount}`,
    `- Safe no-op rate: ${(final.row.safeNoOpRate * 100).toFixed(1)}%`,
    `- Finance exploit count: ${final.row.financeExploitCount}`,
    '',
    '## Interpretation',
    '',
    accepted
      ? 'The optimized hard heuristic passed held-out validation and final gates and can be promoted to the app config.'
      : 'The optimized hard heuristic did not pass the held-out gates. Do not claim hard beats medium until it passes.',
  ].join('\n');

  writeRows('hard', allRows);
  writeFileSync(join(outputDirForMode('hard'), 'best_hard_policy.json'), `${JSON.stringify(bestConfig, null, 2)}\n`);
  writeFileSync(join(outputDirForMode('hard'), 'summary.md'), `${summary}\n`);
  writeFileSync(
    join(ARTIFACT_DIR, accepted ? 'optimized-hard-policy.json' : 'rejected-hard-policy.json'),
    `${JSON.stringify(bestConfig, null, 2)}\n`,
  );

  console.log('Business Empire hard bot optimization complete');
  console.log(
    `validation avg=${validation.row.averageScore.toFixed(2)} vs baseline=${validation.row.baselineAverageScore.toFixed(2)}, scoreLift=${validation.row.scoreLift.toFixed(2)}, rankLift=${validation.row.rankLift.toFixed(2)}, winRateLift=${(validation.row.winRateLift * 100).toFixed(1)}%, invalid=${validation.row.invalidActionCount}, accepted=${accepted ? 'yes' : 'no'}`,
  );
  console.log(
    `final avg=${final.row.averageScore.toFixed(2)} vs baseline=${final.row.baselineAverageScore.toFixed(2)}, scoreLift=${final.row.scoreLift.toFixed(2)}, rankLift=${final.row.rankLift.toFixed(2)}, winRateLift=${(final.row.winRateLift * 100).toFixed(1)}%, invalid=${final.row.invalidActionCount}`,
  );
  console.log(`Artifacts: ${outputDirForMode('hard')}`);
}

function main(): void {
  if (getMode() === 'hard') {
    optimizeHard();
  } else {
    optimizeExpert();
  }
}

main();
