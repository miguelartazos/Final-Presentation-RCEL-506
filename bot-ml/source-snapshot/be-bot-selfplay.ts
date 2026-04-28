import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FALLBACK_CARDS } from '../src/data/fallbackCards';
import type {
  BEBotDifficulty,
  BECardId,
  BECrecerChoice,
  BEFinanzasChoice,
  BEGameState,
  BEPlayer,
  BETalentoChoice,
  Department,
  PlayerConfig,
} from '../src/types/be-game.types';
import { ALL_DEPARTMENTS } from '../src/types/be-game.types';
import { botChooseObjectives, type BotQuarterPlan } from '../src/utils/beBot';
import {
  advanceTurn,
  calculateFinalScore,
  chooseObjectives,
  chooseStartingHand,
  createInitialGameState,
  executeCrecer,
  executeFinanzas,
  executeLanzar,
  executeOportunidades,
  executeTalento,
  getCurrentDepartment,
  getCurrentFocusLevel,
  processBreak,
  respondPendingSale,
  startPlayingPhase,
  startQuarter,
  submitPlan,
} from '../src/utils/beGameLogic';
import {
  HARD_TUNED_BOT_POLICY_CONFIG,
  MEDIUM_BOT_POLICY_CONFIG,
  type BEBotActionCandidate,
  type BEBotPolicyRuntimeOptions,
  chooseBotActionCandidate,
  chooseBotQuarterPlan,
  generateLegalBotActionCandidates,
} from '../src/utils/beBotPolicy';
import {
  BOT_CARD_DATA_VERSION,
  BOT_MODEL_FEATURES,
  BOT_MODEL_VERSION,
  BOT_RULES_VERSION,
  EXPERT_BOT_LINEAR_MODEL,
} from '../src/utils/beBotModel.generated';

export type BenchmarkPolicy = BEBotDifficulty | 'randomDiagnostic';

export type RunRow = {
  policy: BenchmarkPolicy;
  seed: number;
  targetSeat: number;
  targetPlayerId: string;
  winnerId: string;
  targetRank: number;
  targetScore: number;
  targetWon: boolean;
  invalidActionCount: number;
  actionCount: number;
  targetActionCount: number;
  effectiveActionCount: number;
  safeNoOpCount: number;
  financeActionCount: number;
  financeExploitCount: number;
  quartersCompleted: number;
  endTrigger: string;
  runtimeMs: number;
};

export type ActionTraceRow = {
  policy: BenchmarkPolicy;
  seed: number;
  targetSeat: number;
  quarter: number;
  turn: number;
  playerId: string;
  difficulty: string;
  department: string;
  candidateId: string;
  label: string;
  valid: boolean;
  safeNoOp: boolean;
  financeExploitRisk: number;
};

export type SummaryRow = {
  policy: BenchmarkPolicy;
  games: number;
  winRate: number;
  averageRank: number;
  averageScore: number;
  scoreDiffVsSameSeatMedium: number;
  invalidActionCount: number;
  effectiveActionCount: number;
  safeNoOpRate: number;
  financeActionRate: number;
  financeExploitCount: number;
  averageGameLength: number;
  averageRuntimeMs: number;
};

export type SimulationRunOptions = {
  runtimeOptions?: BEBotPolicyRuntimeOptions;
  random?: () => number;
  targetSeat?: number;
};

const OUTPUT_DIR = join(process.cwd(), 'training', 'bot-ml', 'experiments', 'latest');
const REPORT_DIR = join(process.cwd(), 'training', 'bot-ml', 'reports');
const ARTIFACT_DIR = join(process.cwd(), 'training', 'bot-ml', 'artifacts');
const APP_POLICY_ORDER: BEBotDifficulty[] = ['easy', 'medium', 'hard', 'expert'];
export const BENCHMARK_POLICY_ORDER: BenchmarkPolicy[] = ['randomDiagnostic', ...APP_POLICY_ORDER];
const DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606];
const SEATS = [0, 1, 2, 3] as const;
const MAX_QUARTERS = 4;
const MAX_ACTIONS = 260;

function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 0x100000000;
  };
}

function ensureOutputDirs(): void {
  for (const dir of [OUTPUT_DIR, REPORT_DIR, ARTIFACT_DIR]) {
    mkdirSync(dir, { recursive: true });
  }
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

function benchmarkPolicyToAppDifficulty(policy: BenchmarkPolicy): BEBotDifficulty {
  return policy === 'randomDiagnostic' ? 'easy' : policy;
}

function makePlayers(targetPolicy: BenchmarkPolicy, targetSeat: number): PlayerConfig[] {
  return Array.from({ length: 4 }, (_, seat) => {
    const isTarget = seat === targetSeat;
    const difficulty = isTarget ? benchmarkPolicyToAppDifficulty(targetPolicy) : 'medium';
    return {
      id: isTarget ? 'bot-target' : `bot-medium-${seat}`,
      name: isTarget ? `${targetPolicy} target` : `Medium competitor ${seat + 1}`,
      isBot: true,
      botDifficulty: difficulty,
    };
  });
}

function chooseStartingCards(player: BEPlayer, state: BEGameState): BECardId[] {
  return player.hand
    .slice()
    .sort((a, b) => {
      const cardA = state.cardRegistry[a];
      const cardB = state.cardRegistry[b];
      const scoreA = (cardA?.valuationPoints ?? 0) * 2 + (cardA?.income ?? 0) / 1000 - (cardA?.cost ?? 0) / 5000;
      const scoreB = (cardB?.valuationPoints ?? 0) * 2 + (cardB?.income ?? 0) / 1000 - (cardB?.cost ?? 0) / 5000;
      return scoreB - scoreA;
    })
    .slice(0, 4);
}

function chooseNaiveSubset<T>(items: T[], count: number, random: () => number): T[] {
  return items.slice().sort(() => random() - 0.5).slice(0, count);
}

function completeDrafts(state: BEGameState, targetPolicy: BenchmarkPolicy, random: () => number): BEGameState {
  let nextState = state;
  for (const player of nextState.players) {
    const playerPolicy = resolvePlayerPolicy(player, targetPolicy);
    const objectiveIds = playerPolicy === 'randomDiagnostic' || playerPolicy === 'easy'
      ? chooseNaiveSubset(player.secretObjectives, 2, random)
      : botChooseObjectives(player.secretObjectives);
    nextState = chooseObjectives(nextState, player.id, objectiveIds);
  }
  for (const player of nextState.players) {
    const current = nextState.players.find((candidate) => candidate.id === player.id);
    if (current) {
      const playerPolicy = resolvePlayerPolicy(current, targetPolicy);
      const hand = playerPolicy === 'randomDiagnostic' || playerPolicy === 'easy'
        ? chooseNaiveSubset(current.hand, 4, random)
        : chooseStartingCards(current, nextState);
      nextState = chooseStartingHand(nextState, player.id, hand);
    }
  }
  return nextState;
}

function chooseRandomDiagnosticQuarterPlan(random: () => number): BotQuarterPlan {
  const shuffled = ALL_DEPARTMENTS.slice().sort(() => random() - 0.5);
  return {
    plan: shuffled.slice(0, 4),
    hold: shuffled[4],
    temporaryFocusToBuy: 0,
  };
}

function chooseRandomDiagnosticCandidate(
  player: BEPlayer,
  state: BEGameState,
  department: Department,
  focusLevel: 1 | 2 | 3,
  random: () => number,
): BEBotActionCandidate | null {
  const candidates = generateLegalBotActionCandidates(player, state, department, focusLevel);
  if (candidates.length === 0) return null;
  return candidates[Math.floor(random() * candidates.length)] ?? candidates[0];
}

function resolvePlayerPolicy(player: BEPlayer, targetPolicy: BenchmarkPolicy): BenchmarkPolicy {
  return player.id === 'bot-target' ? targetPolicy : 'medium';
}

function submitBotPlans(
  state: BEGameState,
  targetPolicy: BenchmarkPolicy,
  options: SimulationRunOptions = {},
): BEGameState {
  let nextState = state;
  for (const player of nextState.players) {
    const playerPolicy = resolvePlayerPolicy(player, targetPolicy);
    const random = options.random ?? Math.random;
    const plan = playerPolicy === 'randomDiagnostic'
      ? chooseRandomDiagnosticQuarterPlan(random)
      : chooseBotQuarterPlan(
          player,
          nextState,
          playerPolicy,
          random,
          options.runtimeOptions,
        );
    nextState = submitPlan(nextState, player.id, plan.plan, plan.hold, {
      upgradeDepartment: plan.upgradeDepartment,
      temporaryFocusToBuy: plan.temporaryFocusToBuy,
    });
  }
  return startPlayingPhase(nextState);
}

function executeDepartmentAction(
  state: BEGameState,
  player: BEPlayer,
  department: Department,
  targetPolicy: BenchmarkPolicy,
  options: SimulationRunOptions = {},
): { state: BEGameState; valid: boolean; candidateId: string; label: string; financeExploitRisk: number } {
  const focusLevel = getCurrentFocusLevel(player);
  const playerPolicy = resolvePlayerPolicy(player, targetPolicy);
  const random = options.random ?? Math.random;
  const candidate = playerPolicy === 'randomDiagnostic'
    ? chooseRandomDiagnosticCandidate(player, state, department, focusLevel, random)
    : chooseBotActionCandidate(
        player,
        state,
        department,
        focusLevel,
        playerPolicy,
        random,
        options.runtimeOptions,
      );

  if (!candidate) {
    return {
      state,
      valid: true,
      candidateId: 'safe-no-op',
      label: 'No legal action available',
      financeExploitRisk: 0,
    };
  }

  let nextState = state;
  switch (department) {
    case 'OPORTUNIDADES':
      nextState = executeOportunidades(state, player.id, focusLevel, candidate.options);
      break;
    case 'LANZAR':
      if (candidate.options.cardId) {
        nextState = executeLanzar(state, player.id, focusLevel, candidate.options.cardId, {
          placement: candidate.options.placement as { zone: 'city' | 'barrio' | 'digital'; lotIds: string[] } | null | undefined,
          growthCubeTargetId: candidate.options.growthCubeTargetId as BECardId | undefined,
          refreshTargetIds: candidate.options.refreshTargetIds as BECardId[] | undefined,
          useTemporaryFocus: candidate.options.useTemporaryFocus as boolean | undefined,
        });
      }
      break;
    case 'TALENTO':
      if (candidate.options.choice) {
        nextState = executeTalento(state, player.id, focusLevel, {
          choice: candidate.options.choice as BETalentoChoice,
          useTemporaryFocus: candidate.options.useTemporaryFocus as boolean | undefined,
        });
      }
      break;
    case 'CRECER':
      if (candidate.options.choice) {
        nextState = executeCrecer(state, player.id, focusLevel, {
          choice: candidate.options.choice as BECrecerChoice,
          useTemporaryFocus: candidate.options.useTemporaryFocus as boolean | undefined,
        });
      }
      break;
    case 'FINANZAS':
      if (candidate.options.choice) {
        nextState = executeFinanzas(state, player.id, focusLevel, {
          choice: candidate.options.choice as BEFinanzasChoice,
          useTemporaryFocus: candidate.options.useTemporaryFocus as boolean | undefined,
        });
      }
      break;
  }

  return {
    state: nextState,
    valid: nextState !== state && nextState.actionLog.length > state.actionLog.length,
    candidateId: candidate.id,
    label: candidate.label,
    financeExploitRisk: candidate.features.financeExploitRisk ?? 0,
  };
}

export function runGame(
  policy: BenchmarkPolicy,
  seed: number,
  options: SimulationRunOptions = {},
): { row: RunRow; trace: ActionTraceRow[] } {
  const startedAt = Date.now();
  const targetSeat = options.targetSeat ?? 0;
  const random = options.random ?? seededRandom(seed + 999_999 + targetSeat * 10_000);
  let state = createInitialGameState(makePlayers(policy, targetSeat), FALLBACK_CARDS, {
    random: seededRandom(seed),
    gameId: `bot-ml-${policy}-${seed}-seat-${targetSeat}`,
  });
  state = completeDrafts(state, policy, random);
  state = startQuarter(state);

  const trace: ActionTraceRow[] = [];
  let invalidActionCount = 0;
  let actionCount = 0;
  let targetActionCount = 0;
  let effectiveActionCount = 0;
  let safeNoOpCount = 0;
  let financeActionCount = 0;
  let financeExploitCount = 0;

  while (state.currentQuarter <= MAX_QUARTERS && actionCount < MAX_ACTIONS) {
    if (state.phase === 'planning') {
      state = submitBotPlans(state, policy, { ...options, random });
      continue;
    }

    if (state.phase === 'break') {
      state = processBreak(state).state;
      continue;
    }

    if (state.phase === 'endgame' || state.phase === 'gameOver') break;

    if (state.pendingSale) {
      const buyer = state.players.find((player) => player.id === state.pendingSale?.buyerId);
      if (buyer) {
        state = respondPendingSale(state, buyer.id, buyer.cash >= state.pendingSale.minimumPrice);
      }
      continue;
    }

    if (state.phase !== 'playing') break;

    const player = state.players[state.currentPlayerIndex];
    const department = getCurrentDepartment(player);
    if (!department) {
      state = advanceTurn(state);
      continue;
    }

    const result = executeDepartmentAction(state, player, department, policy, { ...options, random });
    if (!result.valid) invalidActionCount += 1;

    const isTarget = player.id === 'bot-target';
    const safeNoOp = result.candidateId === 'safe-no-op';
    if (isTarget) {
      targetActionCount += 1;
      if (safeNoOp) safeNoOpCount += 1;
      else effectiveActionCount += 1;
      if (department === 'FINANZAS' && !safeNoOp) financeActionCount += 1;
      if (result.financeExploitRisk > 0) financeExploitCount += 1;
    }

    trace.push({
      policy,
      seed,
      targetSeat,
      quarter: state.currentQuarter,
      turn: state.currentTurn,
      playerId: player.id,
      difficulty: String(resolvePlayerPolicy(player, policy)),
      department,
      candidateId: result.candidateId,
      label: result.label,
      valid: result.valid,
      safeNoOp,
      financeExploitRisk: result.financeExploitRisk,
    });

    state = result.state;
    actionCount += 1;
    if (!state.pendingSale) state = advanceTurn(state);
  }

  const scored = state.players.map((player) => ({
    player,
    score: calculateFinalScore(player, state.cardRegistry).totalScore,
  }));
  scored.sort((a, b) => b.score - a.score);
  const target = scored.find((entry) => entry.player.id === 'bot-target');
  const targetScore = target?.score ?? 0;
  const targetRank = 1 + scored.filter((entry) => entry.score > targetScore).length;

  return {
    row: {
      policy,
      seed,
      targetSeat,
      targetPlayerId: 'bot-target',
      winnerId: scored[0]?.player.id ?? 'none',
      targetRank,
      targetScore,
      targetWon: scored[0]?.player.id === 'bot-target',
      invalidActionCount,
      actionCount,
      targetActionCount,
      effectiveActionCount,
      safeNoOpCount,
      financeActionCount,
      financeExploitCount,
      quartersCompleted: Math.min(state.currentQuarter, MAX_QUARTERS),
      endTrigger: state.endTrigger ?? 'none',
      runtimeMs: Date.now() - startedAt,
    },
    trace,
  };
}

export function runPolicyBenchmark(
  policy: BenchmarkPolicy,
  seeds: number[],
  options: Omit<SimulationRunOptions, 'targetSeat'> = {},
): { rows: RunRow[]; trace: ActionTraceRow[] } {
  const rows: RunRow[] = [];
  const trace: ActionTraceRow[] = [];
  for (const seed of seeds) {
    for (const targetSeat of SEATS) {
      const result = runGame(policy, seed, { ...options, targetSeat });
      rows.push(result.row);
      trace.push(...result.trace);
    }
  }
  return { rows, trace };
}

export function summarize(rows: RunRow[]): SummaryRow[] {
  const policies = BENCHMARK_POLICY_ORDER.filter((policy) => rows.some((row) => row.policy === policy));
  const mediumBySeedSeat = new Map<string, RunRow>();
  for (const row of rows) {
    if (row.policy === 'medium') mediumBySeedSeat.set(`${row.seed}:${row.targetSeat}`, row);
  }

  return policies.map((policy) => {
    const policyRows = rows.filter((row) => row.policy === policy);
    const count = policyRows.length || 1;
    const targetActionCount = policyRows.reduce((sum, row) => sum + row.targetActionCount, 0);
    const scoreDiffRows = policyRows
      .map((row) => {
        const baseline = mediumBySeedSeat.get(`${row.seed}:${row.targetSeat}`);
        return baseline ? row.targetScore - baseline.targetScore : 0;
      });

    return {
      policy,
      games: policyRows.length,
      winRate: policyRows.filter((row) => row.targetWon).length / count,
      averageRank: policyRows.reduce((sum, row) => sum + row.targetRank, 0) / count,
      averageScore: policyRows.reduce((sum, row) => sum + row.targetScore, 0) / count,
      scoreDiffVsSameSeatMedium: scoreDiffRows.reduce((sum, value) => sum + value, 0) / count,
      invalidActionCount: policyRows.reduce((sum, row) => sum + row.invalidActionCount, 0),
      effectiveActionCount: policyRows.reduce((sum, row) => sum + row.effectiveActionCount, 0),
      safeNoOpRate: targetActionCount > 0
        ? policyRows.reduce((sum, row) => sum + row.safeNoOpCount, 0) / targetActionCount
        : 0,
      financeActionRate: targetActionCount > 0
        ? policyRows.reduce((sum, row) => sum + row.financeActionCount, 0) / targetActionCount
        : 0,
      financeExploitCount: policyRows.reduce((sum, row) => sum + row.financeExploitCount, 0),
      averageGameLength: policyRows.reduce((sum, row) => sum + row.actionCount, 0) / count,
      averageRuntimeMs: policyRows.reduce((sum, row) => sum + row.runtimeMs, 0) / count,
    };
  });
}

function buildSeeds(count: number): number[] {
  if (count <= DEFAULT_SEEDS.length) return DEFAULT_SEEDS.slice(0, count);
  const seeds = [...DEFAULT_SEEDS];
  let current = 707;
  while (seeds.length < count) {
    seeds.push(current);
    current += 101;
  }
  return seeds;
}

function getSeedCountFromArgs(): number {
  const seedFlagIndex = process.argv.findIndex((arg) => arg === '--seeds' || arg === '--seed-count');
  if (seedFlagIndex < 0) return DEFAULT_SEEDS.length;
  const parsed = Number(process.argv[seedFlagIndex + 1]);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : DEFAULT_SEEDS.length;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function writeReports(rows: RunRow[], trace: ActionTraceRow[], seeds: number[]): void {
  ensureOutputDirs();
  const comparison = summarize(rows);
  const medium = comparison.find((row) => row.policy === 'medium');
  const hard = comparison.find((row) => row.policy === 'hard');
  const expert = comparison.find((row) => row.policy === 'expert');
  const randomDiagnostic = comparison.find((row) => row.policy === 'randomDiagnostic');
  const randomOutperformedMedium = Boolean(
    randomDiagnostic &&
    medium &&
    (randomDiagnostic.averageRank <= medium.averageRank ||
      randomDiagnostic.scoreDiffVsSameSeatMedium >= 0),
  );

  const metrics = {
    generatedAt: new Date().toISOString(),
    rulesVersion: BOT_RULES_VERSION,
    cardDataVersion: BOT_CARD_DATA_VERSION,
    modelVersion: BOT_MODEL_VERSION,
    seeds,
    seats: SEATS,
    maxQuarters: MAX_QUARTERS,
    policiesCompared: BENCHMARK_POLICY_ORDER,
    benchmarkStatus: randomOutperformedMedium ? 'diagnostic_failed_random_outperformed_medium' : 'passed',
    comparison,
  };

  const featureImportance = BOT_MODEL_FEATURES.map((feature) => ({
    feature,
    coefficient: EXPERT_BOT_LINEAR_MODEL.coefficients[feature],
    absoluteCoefficient: Math.abs(EXPERT_BOT_LINEAR_MODEL.coefficients[feature]),
    interpretation:
      feature === 'actionVP'
        ? 'The expert bot prefers moves that add valuation points.'
        : feature === 'actionIncome'
          ? 'Higher income improves the action score after normalization.'
          : feature === 'actionCost'
            ? 'Expensive actions are penalized unless their upside is high.'
            : feature === 'placementAffinity'
              ? 'Physical businesses are rewarded for matching the board district.'
              : 'Context feature used by the linear action scorer.',
  })).sort((a, b) => b.absoluteCoefficient - a.absoluteCoefficient);

  const modelReport = {
    modelType: 'linear action scorer',
    modelVersion: BOT_MODEL_VERSION,
    featureNames: BOT_MODEL_FEATURES,
    intercept: EXPERT_BOT_LINEAR_MODEL.intercept,
    coefficients: EXPERT_BOT_LINEAR_MODEL.coefficients,
    validationNote:
      'The benchmark is seat-rotated. Random diagnostic play is tracked separately from app difficulty and is not used as the baseline.',
    topFeatures: featureImportance.slice(0, 5),
  };

  const summary = [
    '# Bot ML Seat-Rotated Benchmark Summary',
    '',
    `Generated: ${metrics.generatedAt}`,
    `Rules version: ${BOT_RULES_VERSION}`,
    `Card data version: ${BOT_CARD_DATA_VERSION}`,
    `Model version: ${BOT_MODEL_VERSION}`,
    `Benchmark status: ${metrics.benchmarkStatus}`,
    '',
    '## Fair Comparison',
    '',
    '| Policy | Games | Win rate | Avg rank | Avg score | Score diff vs same-seat medium | Invalid actions | Effective actions | Safe no-op rate | Finance action rate | Avg game length |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ...comparison.map(
      (row) =>
        `| ${row.policy} | ${row.games} | ${formatPercent(row.winRate)} | ${row.averageRank.toFixed(2)} | ${row.averageScore.toFixed(2)} | ${row.scoreDiffVsSameSeatMedium.toFixed(2)} | ${row.invalidActionCount} | ${row.effectiveActionCount} | ${formatPercent(row.safeNoOpRate)} | ${formatPercent(row.financeActionRate)} | ${row.averageGameLength.toFixed(1)} |`,
    ),
    '',
    '## Interpretation',
    '',
    '- Medium is the naive baseline: rule-based, legal-action-aware, and reproducible.',
    '- Random diagnostic remains in the report only to catch benchmark or game-economy problems.',
    '- ML does not invent moves. The game engine generates legal actions first, then expert ranks those candidates.',
    '- Hard is accepted only if it beats medium on held-out seat-rotated validation.',
    expert && hard && expert.averageRank <= hard.averageRank
      ? '- Expert matched or beat hard by average rank in this benchmark.'
      : '- Expert did not beat hard by average rank in this benchmark, so the tuned heuristic is the stronger current policy.',
    '',
    '## Current Gate Snapshot',
    '',
    `- Hard score diff vs medium: ${(hard?.scoreDiffVsSameSeatMedium ?? 0).toFixed(2)}`,
    `- Expert score diff vs medium: ${(expert?.scoreDiffVsSameSeatMedium ?? 0).toFixed(2)}`,
    `- Random diagnostic outperformed medium: ${randomOutperformedMedium ? 'yes - treat as diagnostic failure' : 'no'}`,
    `- Top expert feature: ${featureImportance[0]?.feature ?? 'n/a'}`,
  ].join('\n');

  const benchmarkAudit = [
    '# Benchmark Audit',
    '',
    '## Why The Previous Result Was Not Trustworthy',
    '',
    'The previous leaderboard compared one target bot against medium opponents from a fixed target seat. It also treated unrestricted random legal play as `easy`, which made the app difficulty look stronger than the taught bots whenever random exploration extended the game or took extra finance actions.',
    '',
    '## What Changed',
    '',
    '- `randomDiagnostic` is now an offline-only policy.',
    '- App `easy` is a novice productive policy, not unrestricted random legal play.',
    '- Every policy is evaluated from every seat on the same seeds.',
    '- Reports include average rank, same-seat score lift, safe no-op rate, finance action rate, and game length.',
    '',
    '## How To Interpret A Random Diagnostic Win',
    '',
    'If `randomDiagnostic` beats medium, the output is a benchmark or game-balance warning. It is not treated as evidence that random play is intelligent, and it is not used as the baseline for the comparison.',
  ].join('\n');

  writeFileSync(join(OUTPUT_DIR, 'metrics.json'), `${JSON.stringify(metrics, null, 2)}\n`);
  writeFileSync(join(OUTPUT_DIR, 'summary.md'), `${summary}\n`);
  writeFileSync(join(OUTPUT_DIR, 'benchmark-audit.md'), `${benchmarkAudit}\n`);
  writeFileSync(
    join(OUTPUT_DIR, 'action_trace.csv'),
    `${toCsv(trace, [
      'policy',
      'seed',
      'targetSeat',
      'quarter',
      'turn',
      'playerId',
      'difficulty',
      'department',
      'candidateId',
      'label',
      'valid',
      'safeNoOp',
      'financeExploitRisk',
    ])}\n`,
  );
  writeFileSync(
    join(OUTPUT_DIR, 'policy_config.json'),
    `${JSON.stringify({
      randomDiagnostic: { version: 'offline-random-legal-action-diagnostic-v1' },
      easy: { version: 'novice-productive-policy-v1' },
      medium: MEDIUM_BOT_POLICY_CONFIG,
      hard: HARD_TUNED_BOT_POLICY_CONFIG,
      expert: {
        modelVersion: BOT_MODEL_VERSION,
        coefficients: EXPERT_BOT_LINEAR_MODEL.coefficients,
      },
    }, null, 2)}\n`,
  );
  writeFileSync(join(OUTPUT_DIR, 'model_report.json'), `${JSON.stringify(modelReport, null, 2)}\n`);
  writeFileSync(
    join(OUTPUT_DIR, 'feature_importance.csv'),
    `${toCsv(featureImportance, ['feature', 'coefficient', 'absoluteCoefficient', 'interpretation'])}\n`,
  );
  writeFileSync(
    join(OUTPUT_DIR, 'before_after_bot_comparison.csv'),
    `${toCsv(comparison, [
      'policy',
      'games',
      'winRate',
      'averageRank',
      'averageScore',
      'scoreDiffVsSameSeatMedium',
      'invalidActionCount',
      'effectiveActionCount',
      'safeNoOpRate',
      'financeActionRate',
      'financeExploitCount',
      'averageGameLength',
      'averageRuntimeMs',
    ])}\n`,
  );
  writeFileSync(
    join(OUTPUT_DIR, 'game_results.csv'),
    `${toCsv(rows, [
      'policy',
      'seed',
      'targetSeat',
      'targetPlayerId',
      'winnerId',
      'targetRank',
      'targetScore',
      'targetWon',
      'invalidActionCount',
      'actionCount',
      'targetActionCount',
      'effectiveActionCount',
      'safeNoOpCount',
      'financeActionCount',
      'financeExploitCount',
      'quartersCompleted',
      'endTrigger',
      'runtimeMs',
    ])}\n`,
  );
  writeFileSync(join(REPORT_DIR, 'bot-ml-summary.md'), `${summary}\n`);
  writeFileSync(join(REPORT_DIR, 'benchmark-audit.md'), `${benchmarkAudit}\n`);
  writeFileSync(join(ARTIFACT_DIR, 'expert-model-report.json'), `${JSON.stringify(modelReport, null, 2)}\n`);
}

function main(): void {
  const rows: RunRow[] = [];
  const trace: ActionTraceRow[] = [];
  const seeds = buildSeeds(getSeedCountFromArgs());

  for (const policy of BENCHMARK_POLICY_ORDER) {
    const result = runPolicyBenchmark(policy, seeds);
    rows.push(...result.rows);
    trace.push(...result.trace);
  }

  writeReports(rows, trace, seeds);
  const comparison = summarize(rows);
  console.log('Business Empire seat-rotated bot benchmark complete');
  for (const row of comparison) {
    console.log(
      `${row.policy}: winRate=${formatPercent(row.winRate)}, avgRank=${row.averageRank.toFixed(2)}, avgScore=${row.averageScore.toFixed(2)}, scoreDiffVsMedium=${row.scoreDiffVsSameSeatMedium.toFixed(2)}, noOpRate=${formatPercent(row.safeNoOpRate)}, invalidActions=${row.invalidActionCount}`,
    );
  }
  console.log(`Artifacts: ${OUTPUT_DIR}`);
}

if (process.argv[1]?.endsWith('be-bot-selfplay.ts')) {
  main();
}
