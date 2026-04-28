import type {
  BEBotDifficulty,
  BEBusinessPlacement,
  BECardDefinition,
  BECardId,
  BECrecerChoice,
  BEFinanzasChoice,
  BEGameState,
  BEPlayer,
  BETalentoChoice,
  Department,
  FocusLevel,
} from '../types/be-game.types';
import { ALL_DEPARTMENTS, BE_CONSTANTS } from '../types/be-game.types';
import { getBusinessSlotCount, validateBusinessPlacement } from './beBoard';
import { botExecuteAction, type BotActionOptions, type BotQuarterPlan } from './beBot';
import {
  executeCrecer,
  executeFinanzas,
  executeLanzar,
  executeOportunidades,
  executeTalento,
} from './beGameLogic';
import type { BEBotLinearModel } from './beBotModel';
import { scoreExpertBotCandidate, scoreExpertBotCandidateWithModel } from './beBotModel';

export type { BEBotDifficulty };

export interface BEBotActionCandidate {
  id: string;
  department: Department;
  label: string;
  options: BotActionOptions;
  features: Record<string, number>;
}

export interface BEBotPolicyConfig {
  version: string;
  weights: {
    cost: number;
    income: number;
    valuationPoints: number;
    placementAffinity: number;
    brand: number;
    handPressure: number;
    temporaryFocus: number;
    projectedNetValue: number;
    remainingQuarters: number;
    cashAfterAction: number;
    staffFit: number;
    launchAffordability: number;
    departmentHasLegalAction: number;
    cashNeed: number;
    loanCapacity: number;
    debtPenaltyProjected: number;
    runwayAfterAction: number;
    financeExploitRisk: number;
  };
  planner: {
    oportunidadesNeed: number;
    lanzarLegalAction: number;
    talentoUnderstaffed: number;
    crecerPortfolio: number;
    finanzasCashNeed: number;
    finanzasLoanNeed: number;
    noLegalActionPenalty: number;
    healthyFinancePenalty: number;
    safeNoOpPenalty: number;
  };
}

export interface BEBotPolicyRuntimeOptions {
  random?: () => number;
  expertModel?: BEBotLinearModel;
  policyConfig?: BEBotPolicyConfig;
}

export const MEDIUM_BOT_POLICY_CONFIG: BEBotPolicyConfig = {
  version: 'medium-rule-based-v1',
  weights: {
    cost: -0.0001,
    income: 0.0002,
    valuationPoints: 0.55,
    placementAffinity: 0.2,
    brand: 0.1,
    handPressure: 0.05,
    temporaryFocus: 0.1,
    projectedNetValue: 0,
    remainingQuarters: 0,
    cashAfterAction: 0,
    staffFit: 0,
    launchAffordability: 0,
    departmentHasLegalAction: 0,
    cashNeed: 0.0002,
    loanCapacity: 0,
    debtPenaltyProjected: -0.2,
    runwayAfterAction: 0.00005,
    financeExploitRisk: -6,
  },
  planner: {
    oportunidadesNeed: 1.3,
    lanzarLegalAction: 1,
    talentoUnderstaffed: 4,
    crecerPortfolio: 0.8,
    finanzasCashNeed: 4,
    finanzasLoanNeed: 2,
    noLegalActionPenalty: -3,
    healthyFinancePenalty: -5,
    safeNoOpPenalty: -4,
  },
};

export const HARD_TUNED_BOT_POLICY_CONFIG: BEBotPolicyConfig = {
  version: 'hard-autoresearch-tuned-candidate-55',
  weights: {
    cost: -0.0001375672028167173,
    income: 0.0007231131071341225,
    valuationPoints: 0.9727641581557691,
    placementAffinity: 0.09170949589461089,
    brand: 0.010159905510954553,
    handPressure: 0.24210789958015086,
    temporaryFocus: -0.009771140245720747,
    projectedNetValue: 8.125643478706457e-7,
    remainingQuarters: -0.10806966824457048,
    cashAfterAction: 0.00007389493066817522,
    staffFit: 0.03275825008749961,
    launchAffordability: -0.005246158223599223,
    departmentHasLegalAction: 0.2232347900979221,
    cashNeed: 0.00012992549946065992,
    loanCapacity: -0.09276266405358913,
    debtPenaltyProjected: -0.6950128221884371,
    runwayAfterAction: 0.00005666283643338829,
    financeExploitRisk: -10.60391654772684,
  },
  planner: {
    oportunidadesNeed: 2.8774854015558957,
    lanzarLegalAction: -0.2784471542108804,
    talentoUnderstaffed: 5.503367797937244,
    crecerPortfolio: 0.12102993186563255,
    finanzasCashNeed: 1.7130059227347374,
    finanzasLoanNeed: 4.420052388450131,
    noLegalActionPenalty: -1.9437292835675182,
    healthyFinancePenalty: -8.81902568996884,
    safeNoOpPenalty: -3.568966284394264,
  },
};

function getPolicyConfig(difficulty: BEBotDifficulty): BEBotPolicyConfig {
  return difficulty === 'hard' || difficulty === 'expert'
    ? HARD_TUNED_BOT_POLICY_CONFIG
    : MEDIUM_BOT_POLICY_CONFIG;
}

function scoreExpertCandidate(
  candidate: BEBotActionCandidate,
  player: BEPlayer,
  state: BEGameState,
  options?: BEBotPolicyRuntimeOptions,
): number {
  return options?.expertModel
    ? scoreExpertBotCandidateWithModel(candidate, player, state, options.expertModel)
    : scoreExpertBotCandidate(candidate, player, state);
}

function shouldUseTemporaryFocus(player: BEPlayer, department: Department): boolean {
  return player.temporaryFocusAvailable > 0 && player.departmentLevels[department] < BE_CONSTANTS.MAX_FOCUS_LEVEL;
}

function hasStateChanged(before: BEGameState, after: BEGameState): boolean {
  return before !== after && after.actionLog.length > before.actionLog.length;
}

function actionCostForCard(card: BECardDefinition, placement?: Pick<BEBusinessPlacement, 'zone' | 'lotIds'> | null): number {
  const locationFee = card.mode === 'Physical' && placement?.zone === 'city' ? BE_CONSTANTS.LOCATION_FEE : 0;
  return card.cost + locationFee;
}

function getRemainingBreaks(state: BEGameState): number {
  return Math.max(1, 5 - state.currentQuarter);
}

function getDebtPrincipal(player: BEPlayer): number {
  return player.activeLoans.reduce((sum, loan) => sum + loan.remainingPrincipal, 0);
}

function getDebtInterest(player: BEPlayer): number {
  return player.activeLoans.reduce((sum, loan) => sum + loan.interestPerBreak, 0);
}

function getLoanCapacity(player: BEPlayer): number {
  const bridgeCount = player.activeLoans.filter((loan) => loan.type === 'bridge').length;
  const growthCount = player.activeLoans.filter((loan) => loan.type === 'growth').length;
  return Math.max(0, BE_CONSTANTS.LOAN_LIMITS.bridge.maxActive - bridgeCount) +
    Math.max(0, BE_CONSTANTS.LOAN_LIMITS.growth.maxActive - growthCount);
}

function baseFinanceFeatures(player: BEPlayer, cashAfterAction = player.cash): Record<string, number> {
  const debtPrincipal = getDebtPrincipal(player);
  return {
    cashNeed: Math.max(0, 3_000 - player.cash),
    loanCapacity: getLoanCapacity(player),
    debtPenaltyProjected: Math.floor(debtPrincipal / 5_000) + player.activeLoans.length * 2,
    runwayAfterAction: cashAfterAction - getDebtInterest(player),
    financeExploitRisk: 0,
  };
}

function financeChoiceIncludesUnneededLoan(player: BEPlayer, choice: BEFinanzasChoice | undefined): boolean {
  if (!choice) return false;
  const needsCash = player.cash < 3_000;
  const hasDebtToManage = player.activeLoans.length > 0;
  if (needsCash || hasDebtToManage) return false;

  const operationTakesLoan = (operation: { kind: string }): boolean => operation.kind === 'take_loan';

  switch (choice.kind) {
    case 'single':
      return operationTakesLoan(choice.operation);
    case 'loan_and_refinance':
      return operationTakesLoan(choice.loanOperation);
    case 'double':
      return choice.operations.some(operationTakesLoan);
    default:
      return false;
  }
}

function candidateIsProductiveForDifficulty(
  player: BEPlayer,
  candidate: BEBotActionCandidate,
  difficulty: BEBotDifficulty,
): boolean {
  if (candidate.department !== 'FINANZAS') return true;

  const choice = candidate.options.choice as BEFinanzasChoice | undefined;
  if (difficulty === 'easy') {
    if (financeChoiceIncludesUnneededLoan(player, choice)) return false;
    if (choice?.kind === 'sell_business') return false;
  }
  return true;
}

function filterCandidatesForDifficulty(
  player: BEPlayer,
  candidates: BEBotActionCandidate[],
  difficulty: BEBotDifficulty,
): BEBotActionCandidate[] {
  return candidates.filter((candidate) => candidateIsProductiveForDifficulty(player, candidate, difficulty));
}

function businessFeatures(
  state: BEGameState,
  card: BECardDefinition,
  player: BEPlayer,
  placement: BEBusinessPlacement | null,
  useTemporaryFocus: boolean,
): Record<string, number> {
  const actionCost = actionCostForCard(card, placement);
  const remainingBreaks = getRemainingBreaks(state);
  const expectedIncomeOverRemainingBreaks = card.income * remainingBreaks;
  const projectedNetValue = expectedIncomeOverRemainingBreaks + card.valuationPoints * 1_000 - actionCost;
  const cashAfterAction = player.cash - actionCost;
  const staffFit = player.employeesReserve >= card.staffMin ? 1 : 0;
  const launchAffordability = cashAfterAction >= 0 ? 1 : 0;

  return {
    actionCost,
    actionIncome: card.income,
    actionVP: card.valuationPoints,
    placementAffinity: placement?.affinityMatched ? 1 : 0,
    brand: player.brand,
    handPressure: Math.max(0, player.hand.length - BE_CONSTANTS.MAX_HAND_SIZE + 2),
    usesTemporaryFocus: useTemporaryFocus ? 1 : 0,
    projectedNetValue,
    remainingQuarters: remainingBreaks,
    cashAfterAction,
    staffFit,
    launchAffordability,
    departmentHasLegalAction: 1,
    ...baseFinanceFeatures(player, cashAfterAction),
  };
}

function getAvailablePlacementRequests(
  state: BEGameState,
  player: BEPlayer,
  card: BECardDefinition,
): Array<Pick<BEBusinessPlacement, 'zone' | 'lotIds'> | null> {
  if (card.mode === 'Digital') return [null];

  const slotCount = getBusinessSlotCount(card);
  const unlockedCityLots = state.board.cityLots.filter((lot) => lot.unlocked && lot.occupiedBy === null);
  const unlockedBarrioLots = (state.board.barrioLots[player.id] ?? []).filter(
    (lot) => lot.unlocked && lot.occupiedBy === null,
  );
  const requests: Array<Pick<BEBusinessPlacement, 'zone' | 'lotIds'>> = [];

  const pushCombinations = (
    zone: 'city' | 'barrio',
    lots: typeof unlockedCityLots,
  ) => {
    if (slotCount === 1) {
      for (const lot of lots) requests.push({ zone, lotIds: [lot.id] });
      return;
    }

    for (let first = 0; first < lots.length; first++) {
      for (let second = first + 1; second < lots.length; second++) {
        requests.push({ zone, lotIds: [lots[first].id, lots[second].id] });
      }
    }
  };

  pushCombinations('city', unlockedCityLots);
  pushCombinations('barrio', unlockedBarrioLots);

  return requests.filter((request) => {
    const result = validateBusinessPlacement({
      state,
      playerId: player.id,
      card,
      requestedPlacement: request,
    });
    return result.valid;
  });
}

function buildLaunchCandidates(
  state: BEGameState,
  player: BEPlayer,
  focusLevel: FocusLevel,
): BEBotActionCandidate[] {
  const useTemporaryFocus = shouldUseTemporaryFocus(player, 'LANZAR');
  const candidates: BEBotActionCandidate[] = [];

  for (const cardId of player.hand) {
    const card = state.cardRegistry[cardId];
    if (!card || card.type !== 'Business') continue;

    for (const request of getAvailablePlacementRequests(state, player, card)) {
      const validation = validateBusinessPlacement({
        state,
        playerId: player.id,
        card,
        requestedPlacement: request,
      });
      if (!validation.valid || !validation.placement) continue;

      const options: BotActionOptions = {
        cardId,
        placement: request,
        useTemporaryFocus,
      };
      const changedState = executeLanzar(state, player.id, focusLevel, cardId, {
        placement: request,
        useTemporaryFocus,
      });
      if (!hasStateChanged(state, changedState)) continue;

      candidates.push({
        id: `LANZAR:${cardId}:${validation.placement.zone}:${validation.placement.lotIds.join('+') || 'digital'}`,
        department: 'LANZAR',
        label: `Launch ${card.name}`,
        options,
        features: businessFeatures(state, card, player, validation.placement, useTemporaryFocus),
      });
    }
  }

  return candidates;
}

function buildLegacyCandidate(
  state: BEGameState,
  player: BEPlayer,
  department: Department,
  focusLevel: FocusLevel,
): BEBotActionCandidate | null {
  if (department === 'LANZAR') return null;

  const options = botExecuteAction(player, state, department, focusLevel);
  let changedState: BEGameState = state;

  switch (department) {
    case 'OPORTUNIDADES':
      changedState = executeOportunidades(state, player.id, focusLevel, options);
      break;
    case 'TALENTO':
      if (!options.choice) return null;
      changedState = executeTalento(state, player.id, focusLevel, {
        choice: options.choice as BETalentoChoice,
        useTemporaryFocus: options.useTemporaryFocus as boolean | undefined,
      });
      break;
    case 'CRECER':
      if (!options.choice) return null;
      changedState = executeCrecer(state, player.id, focusLevel, {
        choice: options.choice as BECrecerChoice,
        useTemporaryFocus: options.useTemporaryFocus as boolean | undefined,
      });
      break;
    case 'FINANZAS':
      if (!options.choice) return null;
      changedState = executeFinanzas(state, player.id, focusLevel, {
        choice: options.choice as BEFinanzasChoice,
        useTemporaryFocus: options.useTemporaryFocus as boolean | undefined,
      });
      break;
  }

  if (!hasStateChanged(state, changedState)) return null;

  return {
    id: `${department}:legacy`,
    department,
    label: `${department} rule-based action`,
    options,
    features: {
      actionCost: 0,
      actionIncome: department === 'FINANZAS' ? 2_000 : 0,
      actionVP: 0,
      placementAffinity: 0,
      brand: player.brand,
      handPressure: Math.max(0, player.hand.length - BE_CONSTANTS.MAX_HAND_SIZE + 2),
      usesTemporaryFocus: options.useTemporaryFocus ? 1 : 0,
      projectedNetValue: department === 'FINANZAS' ? 2_000 : 0,
      remainingQuarters: getRemainingBreaks(state),
      cashAfterAction: player.cash,
      staffFit: 0,
      launchAffordability: 0,
      departmentHasLegalAction: 1,
      ...baseFinanceFeatures(player),
      financeExploitRisk:
        department === 'FINANZAS' && financeChoiceIncludesUnneededLoan(player, options.choice as BEFinanzasChoice | undefined)
          ? 1
          : 0,
    },
  };
}

function buildOportunidadesMarketCandidates(
  state: BEGameState,
  player: BEPlayer,
  focusLevel: FocusLevel,
): BEBotActionCandidate[] {
  if (focusLevel < 2) return [];

  return state.marketRow.flatMap((cardId) => {
    const card = state.cardRegistry[cardId];
    if (!card || card.type !== 'Business') return [];

    const options: BotActionOptions = {
      marketRowPickId: cardId,
      useTemporaryFocus: shouldUseTemporaryFocus(player, 'OPORTUNIDADES'),
    };
    const changedState = executeOportunidades(state, player.id, focusLevel, options);
    if (!hasStateChanged(state, changedState)) return [];

    return [{
      id: `OPORTUNIDADES:market:${cardId}`,
      department: 'OPORTUNIDADES' as const,
      label: `Scout ${card.name}`,
      options,
      features: businessFeatures(state, card, player, null, Boolean(options.useTemporaryFocus)),
    }];
  });
}

export function generateLegalBotActionCandidates(
  player: BEPlayer,
  state: BEGameState,
  department: Department,
  focusLevel: FocusLevel,
): BEBotActionCandidate[] {
  const candidates =
    department === 'LANZAR'
      ? buildLaunchCandidates(state, player, focusLevel)
      : [
          ...(department === 'OPORTUNIDADES'
            ? buildOportunidadesMarketCandidates(state, player, focusLevel)
            : []),
          buildLegacyCandidate(state, player, department, focusLevel),
        ].filter((candidate): candidate is BEBotActionCandidate => candidate !== null);

  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    if (seen.has(candidate.id)) return false;
    seen.add(candidate.id);
    return true;
  });
}

function scoreHeuristicCandidate(candidate: BEBotActionCandidate, player: BEPlayer, config: BEBotPolicyConfig): number {
  const f = candidate.features;
  return (
    (f.actionCost ?? 0) * config.weights.cost +
    (f.actionIncome ?? 0) * config.weights.income +
    (f.actionVP ?? 0) * config.weights.valuationPoints +
    (f.placementAffinity ?? 0) * config.weights.placementAffinity +
    player.brand * config.weights.brand +
    (f.handPressure ?? 0) * config.weights.handPressure +
    (f.usesTemporaryFocus ?? 0) * config.weights.temporaryFocus +
    (f.projectedNetValue ?? 0) * config.weights.projectedNetValue +
    (f.remainingQuarters ?? 0) * config.weights.remainingQuarters +
    (f.cashAfterAction ?? 0) * config.weights.cashAfterAction +
    (f.staffFit ?? 0) * config.weights.staffFit +
    (f.launchAffordability ?? 0) * config.weights.launchAffordability +
    (f.departmentHasLegalAction ?? 0) * config.weights.departmentHasLegalAction +
    (f.cashNeed ?? 0) * config.weights.cashNeed +
    (f.loanCapacity ?? 0) * config.weights.loanCapacity +
    (f.debtPenaltyProjected ?? 0) * config.weights.debtPenaltyProjected +
    (f.runwayAfterAction ?? 0) * config.weights.runwayAfterAction +
    (f.financeExploitRisk ?? 0) * config.weights.financeExploitRisk
  );
}

function scoreDepartmentForPlan(
  player: BEPlayer,
  state: BEGameState,
  department: Department,
  difficulty: BEBotDifficulty,
  options?: BEBotPolicyRuntimeOptions,
): number {
  const focusLevel = player.departmentLevels[department] ?? 1;
  const config = options?.policyConfig ?? getPolicyConfig(difficulty);
  const candidates = filterCandidatesForDifficulty(
    player,
    generateLegalBotActionCandidates(player, state, department, focusLevel),
    difficulty,
  );
  const bestCandidate = chooseHighestScored(candidates, (candidate) => {
    if (difficulty === 'expert') return scoreExpertCandidate(candidate, player, state, options);
    return scoreHeuristicCandidate(candidate, player, config);
  });
  const bestCandidateScore = bestCandidate
    ? difficulty === 'expert'
      ? scoreExpertCandidate(bestCandidate, player, state, options)
      : scoreHeuristicCandidate(bestCandidate, player, config)
    : 0;
  const noLegalAdjustment = candidates.length > 0
    ? config.weights.departmentHasLegalAction
    : config.planner.noLegalActionPenalty + config.planner.safeNoOpPenalty;

  switch (department) {
    case 'OPORTUNIDADES':
      return bestCandidateScore +
        Math.max(0, BE_CONSTANTS.MAX_HAND_SIZE - player.hand.length) * config.planner.oportunidadesNeed +
        noLegalAdjustment +
        1;
    case 'LANZAR':
      return bestCandidateScore + candidates.length * 0.1 + (candidates.length > 0 ? config.planner.lanzarLegalAction : config.planner.noLegalActionPenalty);
    case 'TALENTO': {
      const understaffedCount = player.businesses.filter((business) => {
        const card = state.cardRegistry[business.cardId];
        if (!card) return false;
        const assigned = business.employeesAssigned + (business.hasManager ? BE_CONSTANTS.MANAGER_EQUIVALENCE : 0);
        return assigned < card.staffMin;
      }).length;
      return bestCandidateScore + understaffedCount * config.planner.talentoUnderstaffed + noLegalAdjustment;
    }
    case 'CRECER':
      return bestCandidateScore + player.businesses.length * config.planner.crecerPortfolio + player.boostHand.length * 0.5 + noLegalAdjustment;
    case 'FINANZAS':
      if (player.activeLoans.length === 0 && player.cash >= 12_000) return config.planner.healthyFinancePenalty;
      return bestCandidateScore +
        (player.cash < 3_000
          ? config.planner.finanzasCashNeed
          : player.businesses.length < 3
            ? config.planner.finanzasCashNeed * 0.6
            : 0) +
        player.activeLoans.length * config.planner.finanzasLoanNeed +
        noLegalAdjustment;
  }
}

export function chooseBotQuarterPlan(
  player: BEPlayer,
  state: BEGameState,
  difficulty: BEBotDifficulty = player.botDifficulty ?? 'medium',
  random: () => number = Math.random,
  options?: BEBotPolicyRuntimeOptions,
): BotQuarterPlan {
  if (difficulty === 'easy') {
    const scoredDepartments = ALL_DEPARTMENTS.map((department) => ({
      department,
      score: scoreDepartmentForPlan(player, state, department, difficulty, options) + random() * 0.25,
    })).sort((a, b) => b.score - a.score);
    const plan = scoredDepartments.slice(0, 4).map((entry) => entry.department);
    return {
      plan,
      hold: scoredDepartments[4]?.department ?? 'FINANZAS',
      temporaryFocusToBuy: 0,
    };
  }

  const scoredDepartments = ALL_DEPARTMENTS.map((department) => ({
    department,
    score: scoreDepartmentForPlan(player, state, department, difficulty, options),
  })).sort((a, b) => b.score - a.score);
  const plan = scoredDepartments.slice(0, 4).map((entry) => entry.department);
  const hold = scoredDepartments[4]?.department ?? 'FINANZAS';
  const upgradeDepartment = player.cash >= 18_000
    ? plan.find((department) => player.departmentLevels[department] < BE_CONSTANTS.MAX_FOCUS_LEVEL)
    : undefined;
  const temporaryFocusToBuy = 0;

  return { plan, hold, upgradeDepartment, temporaryFocusToBuy };
}

function chooseHighestScored(
  candidates: BEBotActionCandidate[],
  score: (candidate: BEBotActionCandidate) => number,
): BEBotActionCandidate | null {
  let best: BEBotActionCandidate | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const candidate of candidates) {
    const candidateScore = score(candidate);
    if (candidateScore > bestScore) {
      best = candidate;
      bestScore = candidateScore;
    }
  }

  return best;
}

export function chooseBotActionCandidate(
  player: BEPlayer,
  state: BEGameState,
  department: Department,
  focusLevel: FocusLevel,
  difficulty: BEBotDifficulty = player.botDifficulty ?? 'medium',
  random: () => number = Math.random,
  options?: BEBotPolicyRuntimeOptions,
): BEBotActionCandidate | null {
  const candidates = filterCandidatesForDifficulty(
    player,
    generateLegalBotActionCandidates(player, state, department, focusLevel),
    difficulty,
  );
  if (candidates.length === 0) return null;

  if (difficulty === 'easy') {
    const simpleCandidates = candidates.filter(
      (candidate) => !candidate.id.includes(':market:') && candidate.options.useTemporaryFocus !== true,
    );
    const pool = simpleCandidates.length > 0 ? simpleCandidates : candidates;
    return pool[Math.floor(random() * pool.length)] ?? pool[0];
  }

  if (difficulty === 'expert') {
    return chooseHighestScored(candidates, (candidate) => scoreExpertCandidate(candidate, player, state, options));
  }

  if (
    difficulty === 'medium' ||
    ((difficulty === 'hard' || difficulty === 'expert') && department !== 'LANZAR')
  ) {
    const legacyCandidate = candidates.find((candidate) => candidate.id === `${department}:legacy`);
    if (legacyCandidate) return legacyCandidate;
  }

  const config = options?.policyConfig ?? getPolicyConfig(difficulty);
  return chooseHighestScored(candidates, (candidate) => scoreHeuristicCandidate(candidate, player, config));
}

export function chooseBotAction(
  player: BEPlayer,
  state: BEGameState,
  department: Department,
  focusLevel: FocusLevel,
  difficulty: BEBotDifficulty = player.botDifficulty ?? 'medium',
  random: () => number = Math.random,
  options?: BEBotPolicyRuntimeOptions,
): BotActionOptions {
  return chooseBotActionCandidate(player, state, department, focusLevel, difficulty, random, options)?.options ?? {};
}
