
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
# Statistical analysis
try:
from scipy import stats
from scipy.stats import chi2_contingency
SCIPY_AVAILABLE = True
except ImportError:
SCIPY_AVAILABLE = False
# Plotting
try:
import matplotlib.pyplot as plt
import seaborn as sns
PLOTTING_AVAILABLE = True
except ImportError:
PLOTTING_AVAILABLE = False
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, roc_curve, precision_recall_curve,
confusion_matrix, classification_report
)
class PerformanceEvaluator:
"""Comprehensive performance evaluator for medical image classification"""
def __init__(self, config: Dict[str, Any], output_dir: str):
"""
Initialize performance evaluator
Args:
config: Evaluation configuration
output_dir: Directory to save evaluation outputs
"""
self.config = config
self.output_dir = Path(output_dir)
self.output_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
self.logger = self._setup_logger()
# Evaluation parameters
self.clinical_metrics = config.get('clinical_metrics', True)
self.statistical_tests = config.get('statistical_tests', True)
self.confidence_intervals = config.get('confidence_intervals', True)
self.visualizations = config.get('visualizations', True)
self.logger.info(" Performance Evaluator initialized")
self.logger.info(f" Clinical metrics: {self.clinical_metrics}")
self.logger.info(f" Statistical tests: {self.statistical_tests}")
def _setup_logger(self) -> logging.Logger:
"""Setup logging for evaluation step"""
logger = logging.getLogger('evaluation_step')
logger.setLevel(logging.INFO)
# Remove existing handlers
for handler in logger.handlers[:]:
logger.removeHandler(handler)
# Create handler
log_file = self.output_dir / 'evaluation.log'
handler = logging.FileHandler(log_file)
# Create formatter
formatter = logging.Formatter(
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
return logger
def calculate_clinical_metrics(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
"""
Calculate medical/clinical performance metrics
Args:
cv_results: Cross-validation results
Returns:
Dictionary of clinical metrics
"""
self.logger.info(" Calculating clinical metrics")
mean_metrics = cv_results['mean_metrics']
std_metrics = cv_results['std_metrics']
clinical_metrics = {}
# Basic performance metrics
clinical_metrics['accuracy'] = {
'mean': mean_metrics.get('accuracy', 0.0),
'std': std_metrics.get('accuracy', 0.0),
'clinical_threshold': 0.85 # Common clinical threshold
}
# Sensitivity (Recall) - Critical for medical diagnosis
clinical_metrics['sensitivity'] = {
'mean': mean_metrics.get('recall', mean_metrics.get('sensitivity', 0.0)),
'std': std_metrics.get('recall', std_metrics.get('sensitivity', 0.0)),
'clinical_threshold': 0.90, # High threshold for medical diagnosis
'clinical_importance': 'critical'
}
# Specificity - Important to avoid false positives
clinical_metrics['specificity'] = {
'mean': mean_metrics.get('specificity', 0.0),
'std': std_metrics.get('specificity', 0.0),
'clinical_threshold': 0.85,
'clinical_importance': 'high'
}
# Positive Predictive Value (Precision)
clinical_metrics['ppv'] = {
'mean': mean_metrics.get('precision', mean_metrics.get('ppv', 0.0)),
'std': std_metrics.get('precision', std_metrics.get('ppv', 0.0)),
'clinical_threshold': 0.80,
'clinical_importance': 'high'
}
# Negative Predictive Value
clinical_metrics['npv'] = {
'mean': mean_metrics.get('npv', 0.0),
'std': std_metrics.get('npv', 0.0),
'clinical_threshold': 0.85,
'clinical_importance': 'high'
}
# F1-Score - Overall performance balance
clinical_metrics['f1_score'] = {
'mean': mean_metrics.get('f1', 0.0),
'std': std_metrics.get('f1', 0.0),
'clinical_threshold': 0.80,
'clinical_importance': 'medium'
}
# AUC - Discriminative ability
clinical_metrics['auc'] = {
'mean': mean_metrics.get('auc', 0.0),
'std': std_metrics.get('auc', 0.0),
'clinical_threshold': 0.90,
'clinical_importance': 'high'
}
self.logger.info(" Clinical metrics calculated")
return clinical_metrics
def assess_clinical_acceptability(self, clinical_metrics: Dict[str, Any]) -> Dict[str, Any]:
"""
Assess clinical acceptability of the model
Args:
clinical_metrics: Clinical metrics dictionary
Returns:
Clinical assessment results
"""
self.logger.info("🩺 Assessing clinical acceptability")
# Extract key metrics
accuracy = clinical_metrics['accuracy']['mean']
sensitivity = clinical_metrics['sensitivity']['mean']
specificity = clinical_metrics['specificity']['mean']
ppv = clinical_metrics['ppv']['mean']
npv = clinical_metrics['npv']['mean']
auc = clinical_metrics['auc']['mean']
# Clinical grading system
grade_points = 0
# Accuracy assessment
if accuracy >= 0.90:
grade_points += 4
elif accuracy >= 0.85:
grade_points += 3
elif accuracy >= 0.80:
grade_points += 2
elif accuracy >= 0.75:
grade_points += 1
# Sensitivity assessment (most critical for medical diagnosis)
if sensitivity >= 0.95:
grade_points += 5
elif sensitivity >= 0.90:
grade_points += 4
elif sensitivity >= 0.85:
grade_points += 3
elif sensitivity >= 0.80:
grade_points += 2
elif sensitivity >= 0.75:
grade_points += 1
# Specificity assessment
if specificity >= 0.90:
grade_points += 3
elif specificity >= 0.85:
grade_points += 2
elif specificity >= 0.80:
grade_points += 1
# AUC assessment
if auc >= 0.95:
grade_points += 2
elif auc >= 0.90:
grade_points += 1
# Determine clinical grade
if grade_points >= 12:
grade = "Excellent"
recommendation = "Ready for clinical validation studies"
clinical_readiness = "high"
elif grade_points >= 9:
grade = "Good"
recommendation = "Promising for clinical validation with minor improvements"
clinical_readiness = "medium-high"
elif grade_points >= 6:
grade = "Acceptable"
recommendation = "Requires improvements before clinical validation"
clinical_readiness = "medium"
elif grade_points >= 3:
grade = "Limited"
recommendation = "Significant improvements needed for clinical use"
clinical_readiness = "low"
else:
grade = "Poor"
recommendation = "Not suitable for clinical use in current form"
clinical_readiness = "very-low"
# Specific recommendations based on weak points
recommendations = [recommendation]
if sensitivity < 0.85:
recommendations.append("Improve sensitivity to reduce false negatives (critical for medical diagnosis)")
if specificity < 0.80:
recommendations.append("Improve specificity to reduce false positives and unnecessary interventions")
if accuracy < 0.80:
recommendations.append("Overall accuracy needs improvement for clinical deployment")
if auc < 0.85:
recommendations.append("Model discriminative ability should be enhanced")
# Risk assessment
false_negative_risk = 1.0 - sensitivity
false_positive_risk = 1.0 - specificity
clinical_assessment = {
'grade': grade,
'grade_points': grade_points,
'clinical_readiness': clinical_readiness,
'primary_recommendation': recommendation,
'detailed_recommendations': recommendations,
'risk_assessment': {
'false_negative_risk': float(false_negative_risk),
'false_positive_risk': float(false_positive_risk),
'overall_risk_level': 'high' if false_negative_risk > 0.15 else 'medium' if false_negative_risk > 0.10 else 'low'
},
'clinical_thresholds_met': {
'accuracy_threshold': bool(accuracy >= clinical_metrics['accuracy']['clinical_threshold']),
'sensitivity_threshold': bool(sensitivity >= clinical_metrics['sensitivity']['clinical_threshold']),
'specificity_threshold': bool(specificity >= clinical_metrics['specificity']['clinical_threshold'])
}
}
self.logger.info(f" Clinical grade: {grade}")
self.logger.info(f" Risk level: {clinical_assessment['risk_assessment']['overall_risk_level']}")
return clinical_assessment
def calculate_diagnostic_performance(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
"""
Calculate comprehensive diagnostic performance metrics
Args:
cv_results: Cross-validation results
Returns:
Diagnostic performance metrics
"""
self.logger.info(" Calculating diagnostic performance")
mean_metrics = cv_results['mean_metrics']
confidence_intervals = cv_results.get('confidence_intervals', {})
# Extract metrics with confidence intervals
diagnostic_performance = {}
key_metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'auc']
for metric in key_metrics:
if metric in mean_metrics:
diagnostic_performance[metric] = {
'value': float(mean_metrics[metric]),
'confidence_interval': confidence_intervals.get(metric, (mean_metrics[metric], mean_metrics[metric])),
'clinical_interpretation': self._interpret_metric(metric, mean_metrics[metric])
}
# Calculate derived metrics
if 'sensitivity' in mean_metrics and 'specificity' in mean_metrics:
# Youden's J statistic
youden_j = mean_metrics['sensitivity'] + mean_metrics['specificity'] - 1
diagnostic_performance['youden_j'] = {
'value': float(youden_j),
'interpretation': 'Optimal balance between sensitivity and specificity',
'clinical_interpretation': 'Excellent' if youden_j > 0.8 else 'Good' if youden_j > 0.6 else 'Fair' if youden_j > 0.4 else 'Poor'
}
# Diagnostic Odds Ratio approximation
if all(metric in mean_metrics for metric in ['sensitivity', 'specificity']):
sensitivity = mean_metrics['sensitivity']
specificity = mean_metrics['specificity']
# Avoid division by zero
if sensitivity > 0 and specificity > 0:
dor = (sensitivity / (1 - sensitivity)) / ((1 - specificity) / specificity)
diagnostic_performance['diagnostic_odds_ratio'] = {
'value': float(dor),
'interpretation': 'Higher values indicate better diagnostic performance',
'clinical_interpretation': 'Excellent' if dor > 100 else 'Good' if dor > 25 else 'Fair' if dor > 5 else 'Poor'
}
self.logger.info(" Diagnostic performance calculated")
return diagnostic_performance
def _interpret_metric(self, metric_name: str, value: float) -> str:
"""
Provide clinical interpretation for a metric value
Args:
metric_name: Name of the metric
value: Metric value
Returns:
Clinical interpretation string
"""
if metric_name in ['accuracy', 'f1']:
if value >= 0.90:
return "Excellent clinical performance"
elif value >= 0.85:
return "Good clinical performance"
elif value >= 0.80:
return "Acceptable for clinical use"
elif value >= 0.75:
return "Marginal clinical utility"
else:
return "Insufficient for clinical use"
elif metric_name == 'sensitivity':
if value >= 0.95:
return "Excellent - very low risk of missing cases"
elif value >= 0.90:
return "Good - acceptable false negative rate"
elif value >= 0.85:
return "Acceptable - moderate false negative risk"
elif value >= 0.80:
return "Marginal - higher false negative risk"
else:
return "Poor - unacceptable false negative rate"
elif metric_name == 'specificity':
if value >= 0.90:
return "Excellent - low false positive rate"
elif value >= 0.85:
return "Good - acceptable false positive rate"
elif value >= 0.80:
return "Acceptable - moderate false positive risk"
elif value >= 0.75:
return "Marginal - higher false positive risk"
else:
return "Poor - high false positive rate"
elif metric_name == 'auc':
if value >= 0.95:
return "Outstanding discriminative ability"
elif value >= 0.90:
return "Excellent discriminative ability"
elif value >= 0.85:
return "Good discriminative ability"
elif value >= 0.80:
return "Acceptable discriminative ability"
else:
return "Poor discriminative ability"
return "Metric interpretation not available"
def perform_statistical_analysis(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
"""
Perform comprehensive statistical analysis
Args:
cv_results: Cross-validation results
Returns:
Statistical analysis results
"""
if not self.statistical_tests or not SCIPY_AVAILABLE:
return {'note': 'Statistical analysis requires scipy'}
self.logger.info("Performing statistical analysis")
statistical_results = {}
# Extract fold-level results
fold_results = cv_results.get('fold_results', [])
if not fold_results:
return {'error': 'No fold results available for statistical analysis'}
# Collect metrics across folds
metrics_data = {}
for fold in fold_results:
for metric_name, metric_value in fold['metrics'].items():
if metric_name not in metrics_data:
metrics_data[metric_name] = []
metrics_data[metric_name].append(metric_value)
# Perform normality tests
normality_tests = {}
for metric_name, values in metrics_data.items():
if len(values) >= 3: # Need at least 3 values for normality test
try:
statistic, p_value = stats.shapiro(values)
normality_tests[metric_name] = {
'statistic': float(statistic),
'p_value': float(p_value),
'is_normal': bool(p_value > 0.05)
}
except:
normality_tests[metric_name] = {'error': 'Failed to perform normality test'}
statistical_results['normality_tests'] = normality_tests
# Test for statistical significance vs. baseline
baseline_tests = {}
for metric_name, values in metrics_data.items():
if len(values) >= 3:
# Define baseline based on metric type
if metric_name == 'accuracy':
baseline = 0.5 # Random chance for binary classification
elif metric_name in ['sensitivity', 'specificity', 'precision', 'recall']:
baseline = 0.5
elif metric_name == 'auc':
baseline = 0.5
elif metric_name == 'f1':
baseline = 0.33 # F1 for random classifier
else:
baseline = 0.5
try:
# One-sample t-test
t_stat, p_value = stats.ttest_1samp(values, baseline)
baseline_tests[metric_name] = {
't_statistic': float(t_stat),
'p_value': float(p_value),
'significantly_better': bool(p_value < 0.05 and np.mean(values) > baseline),
'baseline_value': baseline,
'mean_value': float(np.mean(values))
}
except:
baseline_tests[metric_name] = {'error': 'Failed to perform baseline test'}
statistical_results['baseline_comparison'] = baseline_tests
# Calculate effect sizes (Cohen's d)
effect_sizes = {}
for metric_name, values in metrics_data.items():
if len(values) >= 3:
baseline = baseline_tests.get(metric_name, {}).get('baseline_value', 0.5)
try:
cohens_d = (np.mean(values) - baseline) / np.std(values)
effect_sizes[metric_name] = {
'cohens_d': float(cohens_d),
'effect_size_interpretation': self._interpret_effect_size(cohens_d)
}
except:
effect_sizes[metric_name] = {'error': 'Failed to calculate effect size'}
statistical_results['effect_sizes'] = effect_sizes
# Cross-validation stability analysis
stability_analysis = {}
for metric_name, values in metrics_data.items():
cv_coefficient = np.std(values) / np.mean(values) if np.mean(values) > 0 else float('inf')
stability_analysis[metric_name] = {
'cv_coefficient': float(cv_coefficient),
'stability_rating': 'high' if cv_coefficient < 0.1 else 'medium' if cv_coefficient < 0.2 else 'low',
'interpretation': self._interpret_stability(cv_coefficient)
}
statistical_results['stability_analysis'] = stability_analysis
self.logger.info("Statistical analysis completed")
return statistical_results
def _interpret_effect_size(self, cohens_d: float) -> str:
"""Interpret Cohen's d effect size"""
abs_d = abs(cohens_d)
if abs_d < 0.2:
return "negligible"
elif abs_d < 0.5:
return "small"
elif abs_d < 0.8:
return "medium"
else:
return "large"
def _interpret_stability(self, cv_coefficient: float) -> str:
"""Interpret cross-validation coefficient of variation"""
if cv_coefficient < 0.05:
return "Very stable performance across folds"
elif cv_coefficient < 0.10:
return "Stable performance with low variance"
elif cv_coefficient < 0.20:
return "Moderate stability with acceptable variance"
else:
return "High variance - model performance is unstable"
def generate_performance_visualizations(self, cv_results: Dict[str, Any], plots_dir: Path) -> List[str]:
"""
Generate performance visualization plots
Args:
cv_results: Cross-validation results
plots_dir: Directory to save plots
Returns:
List of generated plot file paths
"""
if not self.visualizations or not PLOTTING_AVAILABLE:
return []
self.logger.info("Generating performance visualizations")
plots_dir.mkdir(exist_ok=True)
generated_plots = []
# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
# Extract data
fold_scores = cv_results.get('fold_scores', {})
mean_metrics = cv_results.get('mean_metrics', {})
std_metrics = cv_results.get('std_metrics', {})
# 1. Metrics comparison bar plot
try:
fig, ax = plt.subplots(figsize=(12, 8))
metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']
available_metrics = [m for m in metrics_to_plot if m in mean_metrics]
if available_metrics:
means = [mean_metrics[m] for m in available_metrics]
stds = [std_metrics.get(m, 0) for m in available_metrics]
bars = ax.bar(available_metrics, means, yerr=stds, capsize=5, alpha=0.7)
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Performance Metrics')
ax.set_ylim(0, 1.0)
# Add value labels on bars
for bar, mean_val in zip(bars, means):
height = bar.get_height()
ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
f'{mean_val:.3f}', ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = plots_dir / 'cv_metrics_comparison.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
generated_plots.append(str(plot_path))
except Exception as e:
self.logger.warning(f"Failed to generate metrics comparison plot: {e}")
# 2. Metrics across folds line plot
try:
if fold_scores:
fig, ax = plt.subplots(figsize=(12, 8))
for metric_name, values in fold_scores.items():
if metric_name in ['accuracy', 'sensitivity', 'specificity', 'f1']:
ax.plot(range(1, len(values) + 1), values, marker='o', label=metric_name.capitalize())
ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('Metrics Across Cross-Validation Folds')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plot_path = plots_dir / 'cv_metrics_across_folds.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
generated_plots.append(str(plot_path))
except Exception as e:
self.logger.warning(f"Failed to generate fold metrics plot: {e}")
# 3. Clinical performance radar chart
try:
clinical_metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
available_clinical = [m for m in clinical_metrics if m in mean_metrics]
if len(available_clinical) >= 3:
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2 * np.pi, len(available_clinical), endpoint=False)
values = [mean_metrics[m] for m in available_clinical]
# Close the plot
angles = np.concatenate((angles, [angles[0]]))
values = values + [values[0]]
ax.plot(angles, values, 'o-', linewidth=2, label='Model Performance')
ax.fill(angles, values, alpha=0.25)
# Add clinical threshold line
threshold_values = [0.85] * len(angles)
ax.plot(angles, threshold_values, '--', color='red', linewidth=2, label='Clinical Threshold (0.85)')
ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.capitalize() for m in available_clinical])
ax.set_ylim(0, 1)
ax.set_title('Clinical Performance Radar Chart', size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plot_path = plots_dir / 'clinical_performance_radar.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
generated_plots.append(str(plot_path))
except Exception as e:
self.logger.warning(f"Failed to generate radar chart: {e}")
self.logger.info(f"Generated {len(generated_plots)} visualization plots")
return generated_plots
def evaluate_model_performance(self, cv_results: Dict[str, Any], generate_plots: bool = True,
plots_dir: Optional[Path] = None) -> Dict[str, Any]:
"""
Perform comprehensive model performance evaluation
Args:
cv_results: Cross-validation results
generate_plots: Whether to generate visualization plots
plots_dir: Directory to save plots
Returns:
Complete evaluation results
"""
self.logger.info("Starting comprehensive performance evaluation")
# Calculate clinical metrics
clinical_metrics = self.calculate_clinical_metrics(cv_results)
# Assess clinical acceptability
clinical_assessment = self.assess_clinical_acceptability(clinical_metrics)
# Calculate diagnostic performance
diagnostic_performance = self.calculate_diagnostic_performance(cv_results)
# Perform statistical analysis
statistical_analysis = self.perform_statistical_analysis(cv_results)
# Generate visualizations if requested
generated_plots = []
if generate_plots and plots_dir:
generated_plots = self.generate_performance_visualizations(cv_results, plots_dir)
# Compile evaluation results
evaluation_results = {
'clinical_metrics': clinical_metrics,
'clinical_assessment': clinical_assessment,
'diagnostic_performance': diagnostic_performance,
'statistical_analysis': statistical_analysis,
'cross_validation_summary': {
'n_folds': cv_results.get('n_folds', 0),
'total_samples': cv_results.get('total_samples', 0),
'validation_strategy': cv_results.get('config', {}).get('strategy', 'unknown')
},
'visualizations': {
'plots_generated': len(generated_plots),
'plot_paths': generated_plots
},
'evaluation_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
}
# Save evaluation results
results_path = self.output_dir / 'performance_evaluation.json'
with open(results_path, 'w') as f:
json.dump(evaluation_results, f, indent=2)
# Generate evaluation summary
self._generate_evaluation_summary(evaluation_results)
self.logger.info(f" Performance evaluation completed")
self.logger.info(f" Clinical grade: {clinical_assessment['grade']}")
self.logger.info(f" Results saved to {results_path}")
return evaluation_results
def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]):
"""Generate human-readable evaluation summary"""
summary_lines = []
summary_lines.append("MEDICAL IMAGE CLASSIFICATION - PERFORMANCE EVALUATION")
summary_lines.append("=" * 80)
summary_lines.append("")
# Clinical Assessment
clinical_assessment = evaluation_results['clinical_assessment']
summary_lines.append("CLINICAL ASSESSMENT")
summary_lines.append("-" * 30)
summary_lines.append(f"Clinical Grade: {clinical_assessment['grade']}")
summary_lines.append(f"Clinical Readiness: {clinical_assessment['clinical_readiness']}")
summary_lines.append(f"Primary Recommendation: {clinical_assessment['primary_recommendation']}")
summary_lines.append("")
# Key Metrics
clinical_metrics = evaluation_results['clinical_metrics']
summary_lines.append("KEY PERFORMANCE METRICS")
summary_lines.append("-" * 30)
for metric_name, metric_data in clinical_metrics.items():
mean_val = metric_data['mean']
std_val = metric_data['std']
threshold = metric_data.get('clinical_threshold', 0.0)
status = "" if mean_val >= threshold else ""
summary_lines.append(f"{metric_name.upper():12}: {mean_val:.3f} ± {std_val:.3f} {status}")
summary_lines.append("")
# Risk Assessment
risk_assessment = clinical_assessment.get('risk_assessment', {})
if risk_assessment:
summary_lines.append("RISK ASSESSMENT")
summary_lines.append("-" * 30)
summary_lines.append(f"False Negative Risk: {risk_assessment.get('false_negative_risk', 0):.3f}")
summary_lines.append(f"False Positive Risk: {risk_assessment.get('false_positive_risk', 0):.3f}")
summary_lines.append(f"Overall Risk Level: {risk_assessment.get('overall_risk_level', 'unknown').upper()}")
summary_lines.append("")
# Statistical Analysis Summary
statistical_analysis = evaluation_results.get('statistical_analysis', {})
if 'baseline_comparison' in statistical_analysis:
summary_lines.append("STATISTICAL SIGNIFICANCE")
summary_lines.append("-" * 30)
baseline_tests = statistical_analysis['baseline_comparison']
for metric_name, test_result in baseline_tests.items():
if 'significantly_better' in test_result:
significant = test_result['significantly_better']
p_value = test_result.get('p_value', 1.0)
status = "Significant" if significant else "Not Significant"
summary_lines.append(f"{metric_name.capitalize():12}: {status} (p = {p_value:.6f})")
summary_lines.append("")
# Recommendations
detailed_recommendations = clinical_assessment.get('detailed_recommendations', [])
if detailed_recommendations:
summary_lines.append("RECOMMENDATIONS")
summary_lines.append("-" * 30)
for i, recommendation in enumerate(detailed_recommendations, 1):
summary_lines.append(f"{i}. {recommendation}")
summary_lines.append("")
summary_lines.append("END OF EVALUATION SUMMARY")
summary_lines.append("=" * 80)
# Save summary
summary_text = "\n".join(summary_lines)
summary_path = self.output_dir / 'evaluation_summary.txt'
with open(summary_path, 'w') as f:
f.write(summary_text)
self.logger.info(f" Evaluation summary saved to {summary_path}")
if __name__ == "__main__":
# Example usage
config = {
'clinical_metrics': True,
'statistical_tests': True,
'confidence_intervals': True,
'visualizations': True
}
evaluator = PerformanceEvaluator(config, './evaluation_output')
print(" Performance Evaluator ready for use!")