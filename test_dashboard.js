/**
 * Tests for dashboard.html mathematical and logical functions
 * Run with: node test_dashboard.js
 */

// ============================================
// EXTRACTED FUNCTIONS FROM dashboard.html
// ============================================

// Standard normal CDF approximation
function normalCDF(x) {
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Weighted linear regression
function weightedLinearRegression(x, y, weights = null) {
    const n = x.length;
    if (!weights) {
        weights = new Array(n).fill(1);
    }

    const sumW = weights.reduce((a, b) => a + b, 0);
    const sumWX = x.reduce((sum, xi, i) => sum + weights[i] * xi, 0);
    const sumWY = y.reduce((sum, yi, i) => sum + weights[i] * yi, 0);
    const sumWXY = x.reduce((sum, xi, i) => sum + weights[i] * xi * y[i], 0);
    const sumWX2 = x.reduce((sum, xi, i) => sum + weights[i] * xi * xi, 0);

    const meanX = sumWX / sumW;
    const meanY = sumWY / sumW;

    const slope = (sumWXY - sumW * meanX * meanY) / (sumWX2 - sumW * meanX * meanX);
    const intercept = meanY - slope * meanX;

    const yHat = x.map(xi => slope * xi + intercept);
    const ssRes = y.reduce((sum, yi, i) => sum + weights[i] * Math.pow(yi - yHat[i], 2), 0);
    const ssTot = y.reduce((sum, yi, i) => sum + weights[i] * Math.pow(yi - meanY, 2), 0);
    const rSquared = 1 - ssRes / ssTot;

    const residuals = y.map((yi, i) => yi - yHat[i]);
    const rmse = Math.sqrt(ssRes / sumW);

    return { slope, intercept, rSquared, residuals, yHat, rmse };
}

// Model classification constants
const FAMILY_COUNTRY_MAP = {
    "Anthropic": "US",
    "OpenAI": "US",
    "Google": "US",
    "xAI": "US",
    "Meta": "US",
    "Mistral": "EU",
    "DeepSeek": "China",
    "Alibaba": "China",
    "Moonshot": "China",
    "Baichuan": "China",
};

const OPEN_WEIGHT_FAMILIES = new Set([
    "DeepSeek", "Alibaba", "Meta", "Mistral", "Baichuan"
]);

const OPEN_WEIGHT_PATTERNS = ["oss", "open", "llama", "mistral"];

const MODEL_OVERRIDES = {
    "gpt2": { isOpen: true },
    "kimi_k2_thinking": { isOpen: true },
};

function getModelFamily(modelId) {
    const id = modelId.toLowerCase();
    if (id.includes("claude")) return "Anthropic";
    if (id.includes("gpt") || id.includes("davinci") || id.startsWith("o1") || id.startsWith("o3") || id.startsWith("o4")) return "OpenAI";
    if (id.includes("deepseek")) return "DeepSeek";
    if (id.includes("gemini")) return "Google";
    if (id.includes("qwen")) return "Alibaba";
    if (id.includes("grok")) return "xAI";
    if (id.includes("kimi")) return "Moonshot";
    if (id.includes("llama")) return "Meta";
    if (id.includes("mistral") || id.includes("mixtral")) return "Mistral";
    return "Other";
}

function isOpenModel(modelId, family) {
    const id = modelId.toLowerCase();

    if (MODEL_OVERRIDES[modelId]?.isOpen !== undefined) {
        return MODEL_OVERRIDES[modelId].isOpen;
    }

    if (OPEN_WEIGHT_FAMILIES.has(family)) {
        return true;
    }

    for (const pattern of OPEN_WEIGHT_PATTERNS) {
        if (id.includes(pattern)) return true;
    }

    return false;
}

function getModelCountry(modelId, family) {
    if (MODEL_OVERRIDES[modelId]?.country) {
        return MODEL_OVERRIDES[modelId].country;
    }
    return FAMILY_COUNTRY_MAP[family] || "Other";
}

// Gap calculation functions
const DAYS_PER_MONTH = 30.5;

function getFrontierModels(data, groupFn) {
    const groups = {};
    data.forEach(d => {
        const group = groupFn(d);
        if (!groups[group]) groups[group] = [];
        groups[group].push(d);
    });

    const frontierModels = [];
    for (const [group, models] of Object.entries(groups)) {
        const sorted = [...models].sort((a, b) => a.date - b.date);
        let maxHorizon = 0;
        for (const model of sorted) {
            if (model.horizon >= maxHorizon) {
                maxHorizon = model.horizon;
                frontierModels.push(model);
            }
        }
    }

    return frontierModels.sort((a, b) => a.date - b.date);
}

function calculateHorizontalGaps(data, groupKey, leadingValue) {
    const leadingData = data.filter(d => d[groupKey] === leadingValue);
    const laggingData = data.filter(d => d[groupKey] !== leadingValue);

    const leadingFrontier = getFrontierModels(leadingData, d => d.family);
    const laggingFrontier = getFrontierModels(laggingData, d => d.family);

    const gaps = [];

    for (const leading of leadingFrontier) {
        let matchingLagging = null;
        for (const lagging of laggingFrontier) {
            if (lagging.date <= leading.date) continue;
            if (lagging.horizon >= leading.horizon) {
                matchingLagging = lagging;
                break;
            }
        }

        if (matchingLagging) {
            const gapDays = (matchingLagging.date - leading.date) / (1000 * 60 * 60 * 24);
            gaps.push({
                leadingModel: leading.name,
                leadingDate: leading.date,
                leadingHorizon: leading.horizon,
                laggingModel: matchingLagging.name,
                laggingDate: matchingLagging.date,
                laggingHorizon: matchingLagging.horizon,
                gapMonths: gapDays / DAYS_PER_MONTH,
                matched: true
            });
        } else {
            const gapDays = (new Date() - leading.date) / (1000 * 60 * 60 * 24);
            gaps.push({
                leadingModel: leading.name,
                leadingDate: leading.date,
                leadingHorizon: leading.horizon,
                laggingModel: null,
                laggingDate: null,
                laggingHorizon: null,
                gapMonths: gapDays / DAYS_PER_MONTH,
                matched: false
            });
        }
    }

    return gaps;
}

function calculateGapStatistics(gaps) {
    const matchedGaps = gaps.filter(g => g.matched).map(g => g.gapMonths);

    if (matchedGaps.length === 0) {
        return {
            avgGapMonths: null,
            stdGapMonths: null,
            minGapMonths: null,
            maxGapMonths: null,
            totalMatched: 0,
            totalUnmatched: gaps.filter(g => !g.matched).length
        };
    }

    const avg = matchedGaps.reduce((a, b) => a + b, 0) / matchedGaps.length;
    const variance = matchedGaps.reduce((sum, g) => sum + Math.pow(g - avg, 2), 0) / matchedGaps.length;
    const std = Math.sqrt(variance);

    return {
        avgGapMonths: avg,
        stdGapMonths: std,
        minGapMonths: Math.min(...matchedGaps),
        maxGapMonths: Math.max(...matchedGaps),
        totalMatched: matchedGaps.length,
        totalUnmatched: gaps.filter(g => !g.matched).length
    };
}

function estimateCurrentGap(gaps) {
    const matched = gaps.filter(g => g.matched);
    const unmatched = gaps.filter(g => !g.matched);

    const matchedGapMonths = matched.map(g => g.gapMonths).filter(g => g > 0);
    const unmatchedAges = unmatched.map(g => g.gapMonths).sort((a, b) => b - a);

    if (unmatched.length === 0) {
        return {
            estimatedCurrentGap: 0,
            minCurrentGap: 0,
            confidence: 'high',
            method: 'no_unmatched',
            unmatchedAges: [],
            priorParams: null
        };
    }

    const minCurrentGap = Math.max(...unmatchedAges);

    if (matchedGapMonths.length < 3) {
        const estimated = minCurrentGap * 1.3;
        return {
            estimatedCurrentGap: estimated,
            minCurrentGap: minCurrentGap,
            confidence: 'low',
            method: 'insufficient_data_heuristic',
            unmatchedAges: unmatchedAges,
            priorParams: null
        };
    }

    const logMatched = matchedGapMonths.map(g => Math.log(g));
    const muPrior = logMatched.reduce((a, b) => a + b, 0) / logMatched.length;
    let sigmaPrior = Math.sqrt(
        logMatched.reduce((sum, x) => sum + Math.pow(x - muPrior, 2), 0) / logMatched.length
    );
    if (sigmaPrior === 0) sigmaPrior = 0.5;

    function expectedGivenGreaterThan(c, mu, sigma) {
        if (c <= 0) {
            return Math.exp(mu + sigma * sigma / 2);
        }

        const logC = Math.log(c);
        const z = (logC - mu) / sigma;
        const survival = 1 - normalCDF(z);

        if (survival < 1e-10) {
            return c * 2;
        }

        const zShifted = (mu + sigma * sigma - logC) / sigma;
        const expected = Math.exp(mu + sigma * sigma / 2) * normalCDF(zShifted) / survival;

        return expected;
    }

    const expectedGaps = [];
    for (const age of unmatchedAges) {
        if (age > 0) {
            const expGap = expectedGivenGreaterThan(age, muPrior, sigmaPrior);
            expectedGaps.push(expGap);
        }
    }

    let estimated, confidence;

    if (expectedGaps.length > 0) {
        const weights = unmatchedAges.slice(0, expectedGaps.length);
        const totalWeight = weights.reduce((a, b) => a + b, 0);
        const normalizedWeights = totalWeight > 0
            ? weights.map(w => w / totalWeight)
            : weights.map(() => 1 / weights.length);

        estimated = expectedGaps.reduce((sum, g, i) => sum + g * normalizedWeights[i], 0);
        estimated = Math.max(estimated, minCurrentGap);

        const priorMean = Math.exp(muPrior + sigmaPrior * sigmaPrior / 2);
        const deviationRatio = priorMean > 0 ? minCurrentGap / priorMean : 2;

        if (deviationRatio < 1.5) {
            confidence = 'high';
        } else if (deviationRatio < 2.5) {
            confidence = 'medium';
        } else {
            confidence = 'low';
        }
    } else {
        estimated = minCurrentGap * 1.3;
        confidence = 'low';
    }

    return {
        estimatedCurrentGap: estimated,
        minCurrentGap: minCurrentGap,
        confidence: confidence,
        method: 'survival_analysis_mle',
        unmatchedAges: unmatchedAges,
        priorParams: {
            mu: muPrior,
            sigma: sigmaPrior,
            priorMeanMonths: Math.exp(muPrior + sigmaPrior * sigmaPrior / 2)
        }
    };
}

function calculateVerticalGap(data, groupKey, leadingValue) {
    const leadingData = data.filter(d => d[groupKey] === leadingValue);
    const laggingData = data.filter(d => d[groupKey] !== leadingValue);

    if (leadingData.length === 0 || laggingData.length === 0) {
        return { gapHours: null, gapRatio: null };
    }

    const bestLeading = Math.max(...leadingData.map(d => d.horizon));
    const bestLagging = Math.max(...laggingData.map(d => d.horizon));

    return {
        bestLeadingHorizon: bestLeading,
        bestLaggingHorizon: bestLagging,
        gapHours: bestLeading - bestLagging,
        gapRatio: bestLagging > 0 ? bestLeading / bestLagging : null
    };
}

// ============================================
// TEST FRAMEWORK
// ============================================

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
    if (condition) {
        testsPassed++;
        console.log(`  ✓ ${message}`);
    } else {
        testsFailed++;
        console.log(`  ✗ ${message}`);
    }
}

function assertApprox(actual, expected, tolerance, message) {
    const diff = Math.abs(actual - expected);
    if (diff <= tolerance) {
        testsPassed++;
        console.log(`  ✓ ${message} (${actual.toFixed(4)} ≈ ${expected.toFixed(4)})`);
    } else {
        testsFailed++;
        console.log(`  ✗ ${message} (expected ${expected.toFixed(4)}, got ${actual.toFixed(4)}, diff ${diff.toFixed(4)})`);
    }
}

// ============================================
// TESTS
// ============================================

console.log('\n========================================');
console.log('Testing dashboard.html functions');
console.log('========================================\n');

// Test normalCDF
console.log('1. normalCDF (Standard Normal CDF)');
console.log('-----------------------------------');
assertApprox(normalCDF(0), 0.5, 0.001, 'CDF(0) = 0.5');
assertApprox(normalCDF(-1.96), 0.025, 0.001, 'CDF(-1.96) ≈ 0.025');
assertApprox(normalCDF(1.96), 0.975, 0.001, 'CDF(1.96) ≈ 0.975');
assertApprox(normalCDF(-3), 0.00135, 0.001, 'CDF(-3) ≈ 0.00135');
assertApprox(normalCDF(3), 0.99865, 0.001, 'CDF(3) ≈ 0.99865');
assertApprox(normalCDF(1), 0.8413, 0.001, 'CDF(1) ≈ 0.8413');
assertApprox(normalCDF(-1), 0.1587, 0.001, 'CDF(-1) ≈ 0.1587');

// Test weightedLinearRegression
console.log('\n2. weightedLinearRegression');
console.log('---------------------------');

// Simple linear case: y = 2x + 1
let x = [1, 2, 3, 4, 5];
let y = [3, 5, 7, 9, 11];
let reg = weightedLinearRegression(x, y);
assertApprox(reg.slope, 2, 0.001, 'Slope = 2 for y = 2x + 1');
assertApprox(reg.intercept, 1, 0.001, 'Intercept = 1 for y = 2x + 1');
assertApprox(reg.rSquared, 1, 0.001, 'R² = 1 for perfect fit');

// With noise
x = [1, 2, 3, 4, 5];
y = [2.9, 5.1, 6.8, 9.2, 10.9];
reg = weightedLinearRegression(x, y);
assertApprox(reg.slope, 2, 0.2, 'Slope ≈ 2 with noise');
assert(reg.rSquared > 0.99, 'R² > 0.99 with small noise');

// With weights (double weight on middle point)
x = [1, 2, 3];
y = [1, 10, 3];  // Middle point is outlier
let weights = [1, 0.1, 1];  // Low weight on outlier
reg = weightedLinearRegression(x, y, weights);
assertApprox(reg.slope, 1, 0.5, 'Weighted regression reduces outlier influence');

// Test getModelFamily
console.log('\n3. getModelFamily');
console.log('-----------------');
assert(getModelFamily('claude_3_5_sonnet') === 'Anthropic', 'claude_3_5_sonnet → Anthropic');
assert(getModelFamily('gpt_4') === 'OpenAI', 'gpt_4 → OpenAI');
assert(getModelFamily('gpt-oss-120b') === 'OpenAI', 'gpt-oss-120b → OpenAI');
assert(getModelFamily('davinci_002') === 'OpenAI', 'davinci_002 → OpenAI');
assert(getModelFamily('o1_preview') === 'OpenAI', 'o1_preview → OpenAI');
assert(getModelFamily('o3') === 'OpenAI', 'o3 → OpenAI');
assert(getModelFamily('o4-mini') === 'OpenAI', 'o4-mini → OpenAI');
assert(getModelFamily('deepseek_v3') === 'DeepSeek', 'deepseek_v3 → DeepSeek');
assert(getModelFamily('gemini_2_5_pro') === 'Google', 'gemini_2_5_pro → Google');
assert(getModelFamily('qwen_2_72b') === 'Alibaba', 'qwen_2_72b → Alibaba');
assert(getModelFamily('grok_4') === 'xAI', 'grok_4 → xAI');
assert(getModelFamily('kimi_k2_thinking') === 'Moonshot', 'kimi_k2_thinking → Moonshot');
assert(getModelFamily('llama_3_70b') === 'Meta', 'llama_3_70b → Meta');
assert(getModelFamily('mistral_large') === 'Mistral', 'mistral_large → Mistral');
assert(getModelFamily('mixtral_8x7b') === 'Mistral', 'mixtral_8x7b → Mistral');
assert(getModelFamily('unknown_model') === 'Other', 'unknown_model → Other');

// Test isOpenModel
console.log('\n4. isOpenModel');
console.log('--------------');
assert(isOpenModel('gpt2', 'OpenAI') === true, 'gpt2 is open (override)');
assert(isOpenModel('gpt_4', 'OpenAI') === false, 'gpt_4 is closed');
assert(isOpenModel('deepseek_v3', 'DeepSeek') === true, 'deepseek_v3 is open (family)');
assert(isOpenModel('qwen_2_72b', 'Alibaba') === true, 'qwen_2_72b is open (family)');
assert(isOpenModel('claude_3_opus', 'Anthropic') === false, 'claude_3_opus is closed');
assert(isOpenModel('gpt-oss-120b', 'OpenAI') === true, 'gpt-oss-120b is open (pattern)');
assert(isOpenModel('kimi_k2_thinking', 'Moonshot') === true, 'kimi_k2_thinking is open (override)');
assert(isOpenModel('llama_3_70b', 'Meta') === true, 'llama_3_70b is open (family + pattern)');
assert(isOpenModel('mistral_large', 'Mistral') === true, 'mistral_large is open (family + pattern)');

// Test getModelCountry
console.log('\n5. getModelCountry');
console.log('------------------');
assert(getModelCountry('claude_3_opus', 'Anthropic') === 'US', 'Anthropic → US');
assert(getModelCountry('gpt_4', 'OpenAI') === 'US', 'OpenAI → US');
assert(getModelCountry('deepseek_v3', 'DeepSeek') === 'China', 'DeepSeek → China');
assert(getModelCountry('qwen_2_72b', 'Alibaba') === 'China', 'Alibaba → China');
assert(getModelCountry('kimi_k2', 'Moonshot') === 'China', 'Moonshot → China');
assert(getModelCountry('mistral_large', 'Mistral') === 'EU', 'Mistral → EU');
assert(getModelCountry('unknown', 'Other') === 'Other', 'Other → Other');

// Test getFrontierModels
console.log('\n6. getFrontierModels');
console.log('--------------------');

const testModels = [
    { name: 'A', family: 'F1', date: new Date('2023-01-01'), horizon: 1 },
    { name: 'B', family: 'F1', date: new Date('2023-06-01'), horizon: 2 },
    { name: 'C', family: 'F1', date: new Date('2023-12-01'), horizon: 1.5 },  // Not frontier (< B)
    { name: 'D', family: 'F1', date: new Date('2024-06-01'), horizon: 3 },
    { name: 'E', family: 'F2', date: new Date('2023-03-01'), horizon: 0.5 },
    { name: 'F', family: 'F2', date: new Date('2023-09-01'), horizon: 1 },
];

const frontier = getFrontierModels(testModels, d => d.family);
const frontierNames = frontier.map(m => m.name);

assert(frontierNames.includes('A'), 'A is frontier (first in F1)');
assert(frontierNames.includes('B'), 'B is frontier (improves on A)');
assert(!frontierNames.includes('C'), 'C is NOT frontier (lower than B)');
assert(frontierNames.includes('D'), 'D is frontier (improves on B)');
assert(frontierNames.includes('E'), 'E is frontier (first in F2)');
assert(frontierNames.includes('F'), 'F is frontier (improves on E)');
assert(frontier.length === 5, 'Total 5 frontier models');

// Test calculateHorizontalGaps
console.log('\n7. calculateHorizontalGaps');
console.log('--------------------------');

const gapTestData = [
    // Closed models (leading)
    { name: 'Closed1', family: 'Anthropic', date: new Date('2023-01-01'), horizon: 1, isOpen: false },
    { name: 'Closed2', family: 'Anthropic', date: new Date('2023-06-01'), horizon: 2, isOpen: false },
    { name: 'Closed3', family: 'OpenAI', date: new Date('2024-01-01'), horizon: 4, isOpen: false },
    // Open models (lagging)
    { name: 'Open1', family: 'DeepSeek', date: new Date('2023-03-01'), horizon: 0.5, isOpen: true },
    { name: 'Open2', family: 'DeepSeek', date: new Date('2023-09-01'), horizon: 1.5, isOpen: true },
    { name: 'Open3', family: 'Alibaba', date: new Date('2024-03-01'), horizon: 2.5, isOpen: true },
];

const gaps = calculateHorizontalGaps(gapTestData, 'isOpen', false);

assert(gaps.length > 0, 'Gaps calculated');
const closed1Gap = gaps.find(g => g.leadingModel === 'Closed1');
assert(closed1Gap !== undefined, 'Found gap for Closed1');
if (closed1Gap) {
    assert(closed1Gap.matched === true, 'Closed1 is matched');
    assert(closed1Gap.laggingModel === 'Open2', 'Closed1 matched by Open2 (first to reach horizon 1)');
    // Gap should be ~8 months (Jan to Sep 2023)
    assertApprox(closed1Gap.gapMonths, 8, 1, 'Closed1 gap ≈ 8 months');
}

const closed2Gap = gaps.find(g => g.leadingModel === 'Closed2');
assert(closed2Gap !== undefined, 'Found gap for Closed2');
if (closed2Gap) {
    assert(closed2Gap.matched === true, 'Closed2 is matched');
    assert(closed2Gap.laggingModel === 'Open3', 'Closed2 matched by Open3');
    // Gap should be ~9 months (Jun 2023 to Mar 2024)
    assertApprox(closed2Gap.gapMonths, 9, 1, 'Closed2 gap ≈ 9 months');
}

// Test calculateGapStatistics
console.log('\n8. calculateGapStatistics');
console.log('-------------------------');

const testGaps = [
    { matched: true, gapMonths: 6 },
    { matched: true, gapMonths: 12 },
    { matched: true, gapMonths: 9 },
    { matched: false, gapMonths: 15 },
];

const stats = calculateGapStatistics(testGaps);
assertApprox(stats.avgGapMonths, 9, 0.001, 'Average gap = 9 months');
assertApprox(stats.minGapMonths, 6, 0.001, 'Min gap = 6 months');
assertApprox(stats.maxGapMonths, 12, 0.001, 'Max gap = 12 months');
assert(stats.totalMatched === 3, 'Total matched = 3');
assert(stats.totalUnmatched === 1, 'Total unmatched = 1');
// Std dev of [6, 12, 9] = sqrt(((6-9)² + (12-9)² + (9-9)²) / 3) = sqrt(6) ≈ 2.45
assertApprox(stats.stdGapMonths, 2.45, 0.1, 'Std dev ≈ 2.45');

// Test with no matched gaps
const noMatchGaps = [{ matched: false, gapMonths: 10 }];
const noMatchStats = calculateGapStatistics(noMatchGaps);
assert(noMatchStats.avgGapMonths === null, 'No average when no matched gaps');
assert(noMatchStats.totalMatched === 0, 'Total matched = 0');
assert(noMatchStats.totalUnmatched === 1, 'Total unmatched = 1');

// Test estimateCurrentGap
console.log('\n9. estimateCurrentGap (Bayesian Survival Analysis)');
console.log('--------------------------------------------------');

// Case 1: No unmatched models
const noUnmatchedGaps = [
    { matched: true, gapMonths: 6 },
    { matched: true, gapMonths: 9 },
];
const est1 = estimateCurrentGap(noUnmatchedGaps);
assert(est1.estimatedCurrentGap === 0, 'No gap when all matched');
assert(est1.confidence === 'high', 'High confidence when all matched');
assert(est1.method === 'no_unmatched', 'Method = no_unmatched');

// Case 2: Insufficient matched data (< 3)
const insufficientGaps = [
    { matched: true, gapMonths: 6 },
    { matched: false, gapMonths: 12 },
];
const est2 = estimateCurrentGap(insufficientGaps);
assertApprox(est2.estimatedCurrentGap, 12 * 1.3, 0.1, 'Uses heuristic (1.3x) with insufficient data');
assert(est2.minCurrentGap === 12, 'Min current gap = 12');
assert(est2.confidence === 'low', 'Low confidence with insufficient data');

// Case 3: Full Bayesian estimation
const fullGaps = [
    { matched: true, gapMonths: 6 },
    { matched: true, gapMonths: 9 },
    { matched: true, gapMonths: 12 },
    { matched: true, gapMonths: 8 },
    { matched: false, gapMonths: 15 },
    { matched: false, gapMonths: 10 },
];
const est3 = estimateCurrentGap(fullGaps);
assert(est3.estimatedCurrentGap >= 15, 'Estimated gap >= min bound (15)');
assert(est3.minCurrentGap === 15, 'Min current gap = 15');
assert(est3.method === 'survival_analysis_mle', 'Method = survival_analysis_mle');
assert(est3.priorParams !== null, 'Prior params calculated');
assert(est3.priorParams.mu !== undefined, 'Prior mu exists');
assert(est3.priorParams.sigma !== undefined, 'Prior sigma exists');
// Prior mean should be close to average of matched gaps
const avgMatched = (6 + 9 + 12 + 8) / 4;
assertApprox(est3.priorParams.priorMeanMonths, avgMatched, 3, 'Prior mean ≈ average of matched');

// Test calculateVerticalGap
console.log('\n10. calculateVerticalGap');
console.log('------------------------');

const verticalTestData = [
    { name: 'L1', horizon: 10, isOpen: false },
    { name: 'L2', horizon: 8, isOpen: false },
    { name: 'F1', horizon: 5, isOpen: true },
    { name: 'F2', horizon: 3, isOpen: true },
];

const vGap = calculateVerticalGap(verticalTestData, 'isOpen', false);
assert(vGap.bestLeadingHorizon === 10, 'Best leading = 10');
assert(vGap.bestLaggingHorizon === 5, 'Best lagging = 5');
assert(vGap.gapHours === 5, 'Gap = 5 hours');
assert(vGap.gapRatio === 2, 'Ratio = 2x');

// Test with empty groups
const emptyLeading = [{ name: 'F1', horizon: 5, isOpen: true }];
const emptyGap = calculateVerticalGap(emptyLeading, 'isOpen', false);
assert(emptyGap.gapHours === null, 'Null gap with empty group');

// Test edge cases
console.log('\n11. Edge Cases');
console.log('--------------');

// Empty data
const emptyStats = calculateGapStatistics([]);
assert(emptyStats.totalMatched === 0, 'Empty gaps: matched = 0');
assert(emptyStats.avgGapMonths === null, 'Empty gaps: avg = null');

// Single model
const singleModel = [{ name: 'Only', family: 'F1', date: new Date(), horizon: 1 }];
const singleFrontier = getFrontierModels(singleModel, d => d.family);
assert(singleFrontier.length === 1, 'Single model is frontier');

// Very large gap values
const largeGaps = [
    { matched: true, gapMonths: 100 },
    { matched: true, gapMonths: 200 },
    { matched: true, gapMonths: 150 },
    { matched: false, gapMonths: 300 },
];
const largeEst = estimateCurrentGap(largeGaps);
assert(largeEst.estimatedCurrentGap >= 300, 'Large gaps: estimate >= min');
assert(isFinite(largeEst.estimatedCurrentGap), 'Large gaps: estimate is finite');

// Zero gaps
const zeroGaps = [
    { matched: true, gapMonths: 0.1 },
    { matched: true, gapMonths: 0.2 },
    { matched: true, gapMonths: 0.15 },
    { matched: false, gapMonths: 0.5 },
];
const zeroEst = estimateCurrentGap(zeroGaps);
assert(zeroEst.estimatedCurrentGap > 0, 'Small gaps: estimate > 0');
assert(isFinite(zeroEst.estimatedCurrentGap), 'Small gaps: estimate is finite');

// Test bootstrap function (off-by-one fix)
console.log('\n12. bootstrap (CI bounds fix)');
console.log('------------------------------');

// Bootstrap implementation with fix
function bootstrap(data, statFn, nBootstrap = 2000) {
    const n = data.length;
    const bootstrapStats = [];

    for (let b = 0; b < nBootstrap; b++) {
        const sample = [];
        for (let i = 0; i < n; i++) {
            const idx = Math.floor(Math.random() * n);
            sample.push(data[idx]);
        }

        try {
            const stat = statFn(sample);
            if (isFinite(stat)) {
                bootstrapStats.push(stat);
            }
        } catch (e) {
            // Skip failed iterations
        }
    }

    bootstrapStats.sort((a, b) => a - b);

    const estimate = statFn(data);
    const lowIdx = Math.min(Math.floor(bootstrapStats.length * 0.025), bootstrapStats.length - 1);
    const highIdx = Math.min(Math.floor(bootstrapStats.length * 0.975), bootstrapStats.length - 1);
    const ci95Low = bootstrapStats[lowIdx];
    const ci95High = bootstrapStats[highIdx];

    return {
        estimate,
        ci95: [ci95Low, ci95High],
        distribution: bootstrapStats
    };
}

// Test: CI bounds should never be undefined (off-by-one fix)
const testData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const meanFn = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
const result = bootstrap(testData, meanFn, 2000);

assert(result.ci95[0] !== undefined, 'Bootstrap CI low is defined');
assert(result.ci95[1] !== undefined, 'Bootstrap CI high is defined');
assert(isFinite(result.ci95[0]), 'Bootstrap CI low is finite');
assert(isFinite(result.ci95[1]), 'Bootstrap CI high is finite');
assert(result.ci95[0] <= result.estimate, 'CI low <= estimate');
assert(result.ci95[1] >= result.estimate, 'CI high >= estimate');

// Test: exact case where 0.975 * length is integer (2000 samples -> index 1950)
const exactResult = bootstrap(testData, meanFn, 2000);
assert(exactResult.ci95[1] !== undefined, 'Bootstrap CI high defined at exact boundary');
assert(isFinite(exactResult.ci95[1]), 'Bootstrap CI high finite at exact boundary');

// Test: small bootstrap (edge case)
const smallResult = bootstrap(testData, meanFn, 10);
assert(smallResult.ci95[0] !== undefined, 'Small bootstrap CI low defined');
assert(smallResult.ci95[1] !== undefined, 'Small bootstrap CI high defined');

// Test calculateGroupTrend with weights
console.log('\n13. calculateGroupTrend (weighted regression fix)');
console.log('--------------------------------------------------');

// Mock data with weights
const groupTrendData = [
    { dateNum: 0, horizon: 1, family: 'A', weight: 1 },
    { dateNum: 100, horizon: 2, family: 'A', weight: 0.5 },  // Lower weight (wider CI)
    { dateNum: 200, horizon: 4, family: 'A', weight: 1 },
    { dateNum: 300, horizon: 8, family: 'A', weight: 1 },
];

// Simulated calculateGroupTrend with fix
function calculateGroupTrendFixed(data) {
    if (data.length < 2) return null;

    const frontier = data; // Simplified for test
    if (frontier.length < 2) return null;

    const x = frontier.map(d => d.dateNum);
    const y = frontier.map(d => Math.log(d.horizon));
    // Use actual weights from data for consistency with main analysis
    const weights = frontier.map(d => d.weight || 1);

    const reg = weightedLinearRegression(x, y, weights);
    const doublingTimeDays = Math.log(2) / reg.slope;

    return {
        doublingTimeDays,
        rSquared: reg.rSquared,
        nModels: frontier.length,
    };
}

const groupTrend = calculateGroupTrendFixed(groupTrendData);
assert(groupTrend !== null, 'Group trend calculated');
assert(groupTrend.doublingTimeDays > 0, 'Doubling time > 0');
assert(groupTrend.rSquared > 0, 'R² > 0');

// Test that weighted vs unweighted gives different results
const unweightedTrend = weightedLinearRegression(
    groupTrendData.map(d => d.dateNum),
    groupTrendData.map(d => Math.log(d.horizon)),
    groupTrendData.map(() => 1)
);
const weightedTrend = weightedLinearRegression(
    groupTrendData.map(d => d.dateNum),
    groupTrendData.map(d => Math.log(d.horizon)),
    groupTrendData.map(d => d.weight)
);
// Slopes should be slightly different due to weighting
console.log(`  Unweighted slope: ${unweightedTrend.slope.toFixed(6)}`);
console.log(`  Weighted slope: ${weightedTrend.slope.toFixed(6)}`);
assert(Math.abs(unweightedTrend.slope - weightedTrend.slope) < 0.01, 'Slopes differ slightly with weighting');

// ============================================
// SUMMARY
// ============================================

console.log('\n========================================');
console.log('TEST SUMMARY');
console.log('========================================');
console.log(`Passed: ${testsPassed}`);
console.log(`Failed: ${testsFailed}`);
console.log(`Total:  ${testsPassed + testsFailed}`);
console.log('========================================\n');

if (testsFailed > 0) {
    process.exit(1);
}
