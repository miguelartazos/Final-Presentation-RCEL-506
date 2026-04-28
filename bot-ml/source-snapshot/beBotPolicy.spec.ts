import type { BECardDefinition, BECardId, BEGameState, PlayerConfig } from '../types/be-game.types';
import { botExecuteAction } from './beBot';
import { assertExpertModelFeatureOrder, scoreExpertBotCandidate } from './beBotModel';
import { BOT_MODEL_FEATURES } from './beBotModel.generated';
import {
  HARD_TUNED_BOT_POLICY_CONFIG,
  chooseBotAction,
  chooseBotActionCandidate,
  chooseBotQuarterPlan,
  generateLegalBotActionCandidates,
} from './beBotPolicy';
import { createInitialGameState, executeLanzar } from './beGameLogic';
import { runGame } from '../../scripts/be-bot-selfplay';

function makeCard(overrides: Partial<BECardDefinition> & { id: string }): BECardDefinition {
  return {
    id: overrides.id as BECardId,
    type: 'Business',
    name: overrides.name ?? `Test Card ${overrides.id}`,
    industry: 'Service',
    tier: 'Starter',
    cost: 3_000,
    income: 2_000,
    valuationPoints: 1,
    upkeep: 0,
    mode: 'Physical',
    tempo: 'Estable',
    staffMin: 1,
    staffOpt: null,
    incomeScaled: null,
    trendTrack: '',
    synergyGives: '',
    synergyReceives: '',
    tags: [],
    requirements: [],
    immediateEffect: '',
    ongoingEffect: '',
    timeDelay: 0,
    effort: 1,
    likelihood: 10,
    synergies: [],
    flavorText: '',
    status: 'draft',
    version: 2,
    source: 'human',
    hormoziValue: 20,
    roi: 0.67,
    ...overrides,
  } as BECardDefinition;
}

function makeSetup(): { configs: PlayerConfig[]; cards: BECardDefinition[] } {
  const cards: BECardDefinition[] = [];
  for (let i = 0; i < 24; i++) {
    cards.push(
      makeCard({
        id: `BUS-${i}`,
        name: `Business ${i}`,
        industry: i % 2 === 0 ? 'Service' : 'Tech',
        mode: i % 3 === 0 ? 'Digital' : 'Physical',
        valuationPoints: 1 + (i % 5),
        income: 1_000 + i * 100,
      }),
    );
  }
  for (let i = 0; i < 6; i++) cards.push(makeCard({ id: `STF-${i}`, type: 'Staff', name: `Staff ${i}` }));
  for (let i = 0; i < 8; i++) cards.push(makeCard({ id: `BOO-${i}`, type: 'Boost', name: `Boost ${i}` }));

  return {
    configs: [
      { id: 'bot-1', name: 'Bot 1', isBot: true, botDifficulty: 'medium' },
      { id: 'bot-2', name: 'Bot 2', isBot: true, botDifficulty: 'medium' },
    ],
    cards,
  };
}

function createState(): BEGameState {
  const { configs, cards } = makeSetup();
  return createInitialGameState(configs, cards, {
    random: () => 0.42,
    gameId: 'bot-policy-test',
  });
}

describe('Business Empire bot policies', () => {
  test('medium bot keeps the existing non-launch rule-based action', () => {
    const state = createState();
    const player = state.players[0]!;

    expect(chooseBotAction(player, state, 'OPORTUNIDADES', 1, 'medium')).toEqual(
      botExecuteAction(player, state, 'OPORTUNIDADES', 1),
    );
  });

  test('medium quarter plan remains the legacy planner', () => {
    const state = createState();
    const player = state.players[0]!;

    expect(chooseBotQuarterPlan(player, state, 'medium')).toEqual(expect.objectContaining({
      plan: expect.any(Array),
      hold: expect.any(String),
    }));
  });

  test('easy does not take loans when cash is healthy', () => {
    const state = createState();
    const player = { ...state.players[0]!, cash: 10_000, activeLoans: [] };

    expect(chooseBotAction(player, state, 'FINANZAS', 1, 'easy')).toEqual({});
  });

  test('medium planner avoids no-legal-action departments when productive alternatives exist', () => {
    const state = createState();
    const player = { ...state.players[0]!, cash: 1_000, boostHand: ['BOO-0' as BECardId] };
    const plan = chooseBotQuarterPlan(player, { ...state, players: [player, state.players[1]!] }, 'medium');

    expect(plan.plan).toContain('FINANZAS');
    expect(plan.plan).toContain('CRECER');
    expect(plan.plan).not.toContain('TALENTO');
    expect(plan.hold).toBe('TALENTO');
  });

  test('hard quarter plan prioritizes departments with legal moves', () => {
    const state = createState();
    const player = state.players[0]!;
    const plan = chooseBotQuarterPlan(player, state, 'hard');

    expect(plan.plan).toContain('LANZAR');
    expect(plan.plan).toContain('OPORTUNIDADES');
    expect(plan.plan).toHaveLength(4);
  });

  test('legal launch generator emits executable physical or digital placements', () => {
    const state = createState();
    const player = state.players[0]!;
    const candidates = generateLegalBotActionCandidates(player, state, 'LANZAR', 1);

    expect(candidates.length).toBeGreaterThan(0);
    for (const candidate of candidates.slice(0, 8)) {
      expect(candidate.options.cardId).toBeTruthy();
      const nextState = executeLanzar(state, player.id, 1, candidate.options.cardId as BECardId, {
        placement: candidate.options.placement as { zone: 'city' | 'barrio' | 'digital'; lotIds: string[] } | null | undefined,
        useTemporaryFocus: candidate.options.useTemporaryFocus as boolean | undefined,
      });
      expect(nextState).not.toBe(state);
      expect(nextState.actionLog.length).toBe(state.actionLog.length + 1);
    }
  });

  test.each(['easy', 'medium', 'hard', 'expert'] as const)(
    '%s difficulty returns a legal action or safe no-op',
    (difficulty) => {
      const state = createState();
      const player = { ...state.players[0]!, botDifficulty: difficulty };
      const action = chooseBotAction(player, state, 'LANZAR', 1, difficulty, () => 0);

      if (action.cardId) {
        const nextState = executeLanzar(state, player.id, 1, action.cardId, {
          placement: action.placement as { zone: 'city' | 'barrio' | 'digital'; lotIds: string[] } | null | undefined,
          useTemporaryFocus: action.useTemporaryFocus as boolean | undefined,
        });
        expect(nextState.actionLog.length).toBe(state.actionLog.length + 1);
      } else {
        expect(action).toEqual({});
      }
    },
  );

  test('expert scorer is deterministic for fixed inputs', () => {
    const state = createState();
    const player = state.players[0]!;
    const candidate = chooseBotActionCandidate(player, state, 'LANZAR', 1, 'expert')!;

    expect(scoreExpertBotCandidate(candidate, player, state)).toBe(scoreExpertBotCandidate(candidate, player, state));
  });

  test('generated model artifact exposes the declared feature order', () => {
    expect(() => assertExpertModelFeatureOrder()).not.toThrow();
    expect(BOT_MODEL_FEATURES).toEqual([
      'cash',
      'brand',
      'netProfit',
      'businessCount',
      'handCount',
      'actionCost',
      'actionIncome',
      'actionVP',
      'placementAffinity',
      'usesTemporaryFocus',
    ]);
  });

  test('hard policy config declares every optimized weight and planner parameter', () => {
    expect(Object.keys(HARD_TUNED_BOT_POLICY_CONFIG.weights).sort()).toEqual([
      'brand',
      'cashAfterAction',
      'cashNeed',
      'cost',
      'debtPenaltyProjected',
      'departmentHasLegalAction',
      'financeExploitRisk',
      'handPressure',
      'income',
      'launchAffordability',
      'loanCapacity',
      'placementAffinity',
      'projectedNetValue',
      'remainingQuarters',
      'runwayAfterAction',
      'staffFit',
      'temporaryFocus',
      'valuationPoints',
    ]);
    expect(Object.keys(HARD_TUNED_BOT_POLICY_CONFIG.planner).sort()).toEqual([
      'crecerPortfolio',
      'finanzasCashNeed',
      'finanzasLoanNeed',
      'healthyFinancePenalty',
      'lanzarLegalAction',
      'noLegalActionPenalty',
      'oportunidadesNeed',
      'safeNoOpPenalty',
      'talentoUnderstaffed',
    ]);
  });

  test('finance candidates expose deterministic audit features', () => {
    const state = createState();
    const player = { ...state.players[0]!, cash: 10_000, activeLoans: [] };
    const candidate = generateLegalBotActionCandidates(player, state, 'FINANZAS', 1)[0]!;

    expect(candidate.features).toEqual(expect.objectContaining({
      cashNeed: 0,
      loanCapacity: 3,
      debtPenaltyProjected: 0,
      runwayAfterAction: 10_000,
      financeExploitRisk: 1,
    }));
  });

  test('hard self-play is deterministic for a fixed seed', () => {
    const first = runGame('hard', 12_345, { targetSeat: 2 }).row;
    const second = runGame('hard', 12_345, { targetSeat: 2 }).row;

    expect(second).toEqual(expect.objectContaining({
      winnerId: first.winnerId,
      targetRank: first.targetRank,
      targetScore: first.targetScore,
      targetWon: first.targetWon,
      invalidActionCount: first.invalidActionCount,
      actionCount: first.actionCount,
      safeNoOpCount: first.safeNoOpCount,
      quartersCompleted: first.quartersCompleted,
    }));
  });

  test('round-robin benchmark includes each target seat', () => {
    const seeds = [101, 202];
    const seats = seeds.flatMap((seed) => [0, 1, 2, 3].map((targetSeat) => runGame('medium', seed, { targetSeat }).row.targetSeat));

    expect(seats).toEqual([0, 1, 2, 3, 0, 1, 2, 3]);
  });

  test('hard self-play smoke suite has zero invalid actions', () => {
    const seeds = [101, 202, 303, 404, 505];
    const invalidActions = seeds.reduce((sum, seed) => sum + runGame('hard', seed).row.invalidActionCount, 0);

    expect(invalidActions).toBe(0);
  });
});
