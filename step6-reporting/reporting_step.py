
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
# Template and visualization libraries
try:
import matplotlib.pyplot as plt
import seaborn as sns
PLOTTING_AVAILABLE = True
except ImportError:
PLOTTING_AVAILABLE = False
try:
from jinja2 import Template
JINJA2_AVAILABLE = True
except ImportError:
JINJA2_AVAILABLE = False
class ReportGenerator:
"""Comprehensive report generator for medical image classification"""
def __init__(self, config: Dict[str, Any], output_dir: str):
"""
Initialize report generator
Args:
config: Reporting configuration
output_dir: Directory to save reports
"""
self.config = config
self.output_dir = Path(output_dir)
self.output_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
self.logger = self._setup_logger()
# Reporting parameters
self.generate_plots = config.get('generate_plots', True)
self.clinical_assessment = config.get('clinical_assessment', True)
self.detailed_report = config.get('detailed_report', True)
self.summary_dashboard = config.get('summary_dashboard', True)
self.logger.info("Report Generator initialized")
self.logger.info(f"Generate plots: {self.generate_plots}")
self.logger.info(f" Clinical assessment: {self.clinical_assessment}")
def _setup_logger(self) -> logging.Logger:
"""Setup logging for reporting step"""
logger = logging.getLogger('reporting_step')
logger.setLevel(logging.INFO)
# Remove existing handlers
for handler in logger.handlers[:]:
logger.removeHandler(handler)
# Create handler
log_file = self.output_dir / 'reporting.log'
handler = logging.FileHandler(log_file)
# Create formatter
formatter = logging.Formatter(
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
return logger
def generate_executive_summary(self, results: Dict[str, Any]) -> str:
"""
Generate executive summary for stakeholders
Args:
results: Complete pipeline results
Returns:
Executive summary text
"""
pipeline_info = results.get('pipeline_info', {})
evaluation_results = results.get('evaluation', {})
validation_results = results.get('validation', {})
# Extract key information
experiment_name = pipeline_info.get('id', 'Medical Classification Experiment')
clinical_grade = evaluation_results.get('clinical_assessment', {}).get('grade', 'Unknown')
mean_accuracy = validation_results.get('mean_metrics', {}).get('accuracy', 0.0)
summary_lines = []
summary_lines.append("EXECUTIVE SUMMARY")
summary_lines.append("=" * 50)
summary_lines.append("")
# Project Overview
summary_lines.append("PROJECT OVERVIEW")
summary_lines.append("-" * 20)
summary_lines.append(f"Experiment: {experiment_name}")
summary_lines.append("Objective: Develop and validate an AI system for medical image classification")
summary_lines.append("Methodology: Improved Heap-Based Optimization with cross-validation")
summary_lines.append("")
# Key Results
summary_lines.append("KEY RESULTS")
summary_lines.append("-" * 20)
summary_lines.append(f"• Clinical Grade: {clinical_grade}")
summary_lines.append(f"• Overall Accuracy: {mean_accuracy:.1%}")
# Extract other key metrics
clinical_metrics = evaluation_results.get('clinical_metrics', {})
if 'sensitivity' in clinical_metrics:
sensitivity = clinical_metrics['sensitivity'].get('mean', 0)
summary_lines.append(f"• Sensitivity: {sensitivity:.1%}")
if 'specificity' in clinical_metrics:
specificity = clinical_metrics['specificity'].get('mean', 0)
summary_lines.append(f"• Specificity: {specificity:.1%}")
summary_lines.append("")
# Clinical Assessment
clinical_assessment = evaluation_results.get('clinical_assessment', {})
if clinical_assessment:
summary_lines.append("CLINICAL ASSESSMENT")
summary_lines.append("-" * 20)
readiness = clinical_assessment.get('clinical_readiness', 'unknown')
recommendation = clinical_assessment.get('primary_recommendation', 'No recommendation available')
summary_lines.append(f"• Clinical Readiness: {readiness.title()}")
summary_lines.append(f"• Recommendation: {recommendation}")
# Risk assessment
risk_assessment = clinical_assessment.get('risk_assessment', {})
if risk_assessment:
risk_level = risk_assessment.get('overall_risk_level', 'unknown')
summary_lines.append(f"• Risk Level: {risk_level.title()}")
summary_lines.append("")
# Business Impact
summary_lines.append("BUSINESS IMPACT")
summary_lines.append("-" * 20)
if clinical_grade in ['Excellent', 'Good']:
summary_lines.append("• High potential for clinical deployment")
summary_lines.append("• Recommended for clinical validation studies")
summary_lines.append("• Strong ROI potential through improved diagnosis")
elif clinical_grade == 'Acceptable':
summary_lines.append("• Moderate potential with further development")
summary_lines.append("• Additional optimization recommended")
summary_lines.append("• Pilot study recommended before full deployment")
else:
summary_lines.append("• Requires significant improvement before deployment")
summary_lines.append("• Additional research and development needed")
summary_lines.append("• Not recommended for clinical use in current form")
summary_lines.append("")
# Next Steps
summary_lines.append("NEXT STEPS")
summary_lines.append("-" * 20)
detailed_recommendations = clinical_assessment.get('detailed_recommendations', [])
if detailed_recommendations:
for i, rec in enumerate(detailed_recommendations[:3], 1): # Top 3 recommendations
summary_lines.append(f"{i}. {rec}")
else:
summary_lines.append("1. Review detailed technical report")
summary_lines.append("2. Consult with medical experts")
summary_lines.append("3. Plan next phase of development")
summary_lines.append("")
summary_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
return "\n".join(summary_lines)
def generate_clinical_report(self, results: Dict[str, Any]) -> str:
"""
Generate clinical report for medical professionals
Args:
results: Complete pipeline results
Returns:
Clinical report text
"""
evaluation_results = results.get('evaluation', {})
validation_results = results.get('validation', {})
report_lines = []
report_lines.append("CLINICAL VALIDATION REPORT")
report_lines.append("Medical Image Classification System")
report_lines.append("=" * 60)
report_lines.append("")
# Study Design
report_lines.append("STUDY DESIGN")
report_lines.append("-" * 30)
validation_config = validation_results.get('config', {})
report_lines.append(f"Study Type: Cross-sectional validation study")
report_lines.append(f"Validation Method: {validation_config.get('method', 'k-fold cross-validation')}")
report_lines.append(f"Number of Folds: {validation_results.get('n_folds', 'Unknown')}")
report_lines.append(f"Total Samples: {validation_results.get('total_samples', 'Unknown')}")
report_lines.append(f"Validation Strategy: {validation_config.get('strategy', 'Unknown')}")
report_lines.append("")
# Clinical Performance Metrics
report_lines.append("CLINICAL PERFORMANCE METRICS")
report_lines.append("-" * 30)
clinical_metrics = evaluation_results.get('clinical_metrics', {})
confidence_intervals = validation_results.get('confidence_intervals', {})
key_clinical_metrics = [
('accuracy', 'Overall Accuracy'),
('sensitivity', 'Sensitivity (True Positive Rate)'),
('specificity', 'Specificity (True Negative Rate)'),
('ppv', 'Positive Predictive Value'),
('npv', 'Negative Predictive Value'),
('f1_score', 'F1-Score'),
('auc', 'Area Under ROC Curve')
]
for metric_key, metric_name in key_clinical_metrics:
if metric_key in clinical_metrics:
metric_data = clinical_metrics[metric_key]
mean_val = metric_data.get('mean', 0)
std_val = metric_data.get('std', 0)
threshold = metric_data.get('clinical_threshold', 0)
# Get confidence interval
ci = confidence_intervals.get(metric_key, (mean_val, mean_val))
status = " PASS" if mean_val >= threshold else " FAIL"
report_lines.append(f"{metric_name:30}: {mean_val:.3f} ± {std_val:.3f}")
report_lines.append(f"{'95% Confidence Interval':30}: [{ci[0]:.3f}, {ci[1]:.3f}]")
report_lines.append(f"{'Clinical Threshold':30}: {threshold:.3f} {status}")
report_lines.append("")
# Diagnostic Performance Assessment
diagnostic_performance = evaluation_results.get('diagnostic_performance', {})
if diagnostic_performance:
report_lines.append("DIAGNOSTIC PERFORMANCE ASSESSMENT")
report_lines.append("-" * 30)
for metric_name, metric_data in diagnostic_performance.items():
if isinstance(metric_data, dict) and 'clinical_interpretation' in metric_data:
value = metric_data.get('value', 0)
interpretation = metric_data.get('clinical_interpretation', 'No interpretation')
report_lines.append(f"{metric_name.replace('_', ' ').title():25}: {value:.3f}")
report_lines.append(f"{'Clinical Interpretation':25}: {interpretation}")
report_lines.append("")
# Statistical Significance
statistical_analysis = evaluation_results.get('statistical_analysis', {})
if 'baseline_comparison' in statistical_analysis:
report_lines.append("STATISTICAL SIGNIFICANCE ANALYSIS")
report_lines.append("-" * 30)
baseline_tests = statistical_analysis['baseline_comparison']
for metric_name, test_result in baseline_tests.items():
if 'significantly_better' in test_result:
p_value = test_result.get('p_value', 1.0)
significant = test_result['significantly_better']
baseline = test_result.get('baseline_value', 0)
result = "Statistically Significant" if significant else "Not Statistically Significant"
report_lines.append(f"{metric_name.replace('_', ' ').title():20}: {result}")
report_lines.append(f"{'p-value':20}: {p_value:.6f}")
report_lines.append(f"{'Baseline Comparison':20}: vs {baseline:.3f}")
report_lines.append("")
# Clinical Risk Assessment
clinical_assessment = evaluation_results.get('clinical_assessment', {})
risk_assessment = clinical_assessment.get('risk_assessment', {})
if risk_assessment:
report_lines.append("CLINICAL RISK ASSESSMENT")
report_lines.append("-" * 30)
fn_risk = risk_assessment.get('false_negative_risk', 0)
fp_risk = risk_assessment.get('false_positive_risk', 0)
overall_risk = risk_assessment.get('overall_risk_level', 'unknown')
report_lines.append(f"False Negative Risk: {fn_risk:.3f} ({fn_risk*100:.1f}%)")
report_lines.append(f" Clinical Impact: Risk of missing true positive cases")
report_lines.append("")
report_lines.append(f"False Positive Risk: {fp_risk:.3f} ({fp_risk*100:.1f}%)")
report_lines.append(f" Clinical Impact: Risk of unnecessary interventions")
report_lines.append("")
report_lines.append(f"Overall Risk Level: {overall_risk.upper()}")
report_lines.append("")
# Clinical Recommendations
report_lines.append("CLINICAL RECOMMENDATIONS")
report_lines.append("-" * 30)
grade = clinical_assessment.get('grade', 'Unknown')
primary_rec = clinical_assessment.get('primary_recommendation', 'No recommendation available')
detailed_recs = clinical_assessment.get('detailed_recommendations', [])
report_lines.append(f"Clinical Grade: {grade}")
report_lines.append(f"Primary Recommendation: {primary_rec}")
report_lines.append("")
if detailed_recs:
report_lines.append("Detailed Recommendations:")
for i, rec in enumerate(detailed_recs, 1):
report_lines.append(f"{i}. {rec}")
report_lines.append("")
# Regulatory Considerations
report_lines.append("REGULATORY CONSIDERATIONS")
report_lines.append("-" * 30)
if grade in ['Excellent', 'Good']:
report_lines.append("• Model shows promise for regulatory submission")
report_lines.append("• Recommend FDA Pre-Submission meeting")
report_lines.append("• Consider 510(k) pathway for medical device classification")
report_lines.append("• Ensure compliance with ISO 13485 and ISO 14155")
else:
report_lines.append("• Additional validation required before regulatory submission")
report_lines.append("• Consider expanded clinical studies")
report_lines.append("• Address performance limitations identified in this study")
report_lines.append("")
report_lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
report_lines.append("This report is for research purposes only and does not constitute medical advice.")
return "\n".join(report_lines)
def generate_technical_report(self, results: Dict[str, Any]) -> str:
"""
Generate detailed technical report for researchers
Args:
results: Complete pipeline results
Returns:
Technical report text
"""
report_lines = []
report_lines.append("TECHNICAL VALIDATION REPORT")
report_lines.append("Medical Image Classification with Improved HBO")
report_lines.append("=" * 70)
report_lines.append("")
# Abstract
report_lines.append("ABSTRACT")
report_lines.append("-" * 20)
evaluation_results = results.get('evaluation', {})
validation_results = results.get('validation', {})
optimization_results = results.get('optimization', {})
clinical_grade = evaluation_results.get('clinical_assessment', {}).get('grade', 'Unknown')
mean_accuracy = validation_results.get('mean_metrics', {}).get('accuracy', 0)
hbo_improvement = optimization_results.get('improvement', 0)
report_lines.append("This study presents a comprehensive validation of a medical image classification")
report_lines.append("system using an improved Heap-Based Optimization (HBO) algorithm. The system")
report_lines.append(f"achieved a clinical grade of '{clinical_grade}' with an overall accuracy of")
report_lines.append(f"{mean_accuracy:.1%} and HBO improvement of {hbo_improvement:.1%} over baseline methods.")
report_lines.append("")
# Methods
report_lines.append("METHODS")
report_lines.append("-" * 20)
# Preprocessing
preprocessing_results = results.get('preprocessing', {})
if preprocessing_results:
summary = preprocessing_results.get('data_summary', {})
report_lines.append("Data Preprocessing:")
report_lines.append(f"• Total images processed: {summary.get('total_images', 'Unknown')}")
report_lines.append(f"• Success rate: {summary.get('processing_success_rate', 0):.1%}")
report_lines.append(f"• Quality assessment: {summary.get('average_quality_score', 0):.3f}")
report_lines.append("")
# Model Training
training_results = results.get('training', {})
if training_results:
model_arch = training_results.get('model_architecture', 'Unknown')
training_time = training_results.get('training_time', 0)
report_lines.append("Model Training:")
report_lines.append(f"• Architecture: {model_arch}")
report_lines.append(f"• Training time: {training_time:.1f} seconds")
report_lines.append("")
# HBO Optimization
if optimization_results:
best_params = optimization_results.get('best_hyperparameters', {})
optimization_time = optimization_results.get('execution_time', 0)
report_lines.append("HBO Optimization:")
report_lines.append(f"• Execution time: {optimization_time:.1f} seconds")
report_lines.append(f"• Performance improvement: {hbo_improvement:.1%}")
if best_params:
report_lines.append("• Optimized parameters:")
for param, value in best_params.items():
if isinstance(value, float):
report_lines.append(f" - {param}: {value:.4f}")
else:
report_lines.append(f" - {param}: {value}")
report_lines.append("")
# Cross-Validation
if validation_results:
cv_config = validation_results.get('config', {})
report_lines.append("Cross-Validation:")
report_lines.append(f"• Method: {cv_config.get('method', 'Unknown')}")
report_lines.append(f"• Folds: {validation_results.get('n_folds', 'Unknown')}")
report_lines.append(f"• Strategy: {cv_config.get('strategy', 'Unknown')}")
report_lines.append(f"• Total validation time: {validation_results.get('total_validation_time', 0):.1f} seconds")
report_lines.append("")
# Results
report_lines.append("RESULTS")
report_lines.append("-" * 20)
# Performance metrics table
mean_metrics = validation_results.get('mean_metrics', {})
std_metrics = validation_results.get('std_metrics', {})
if mean_metrics:
report_lines.append("Performance Metrics (Mean ± Standard Deviation):")
report_lines.append("")
for metric_name in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']:
if metric_name in mean_metrics:
mean_val = mean_metrics[metric_name]
std_val = std_metrics.get(metric_name, 0)
report_lines.append(f" {metric_name.capitalize():12}: {mean_val:.4f} ± {std_val:.4f}")
report_lines.append("")
# Statistical Analysis
statistical_analysis = evaluation_results.get('statistical_analysis', {})
if statistical_analysis:
report_lines.append("Statistical Analysis:")
# Normality tests
normality_tests = statistical_analysis.get('normality_tests', {})
if normality_tests:
normal_metrics = [name for name, test in normality_tests.items() if isinstance(test, dict) and test.get('is_normal', False)]
report_lines.append(f"• Normally distributed metrics: {', '.join(normal_metrics) if normal_metrics else 'None'}")
# Baseline comparisons
baseline_tests = statistical_analysis.get('baseline_comparison', {})
if baseline_tests:
significant_metrics = [name for name, test in baseline_tests.items()
if isinstance(test, dict) and test.get('significantly_better', False)]
report_lines.append(f"• Significantly better than baseline: {', '.join(significant_metrics) if significant_metrics else 'None'}")
# Stability analysis
stability_analysis = statistical_analysis.get('stability_analysis', {})
if stability_analysis:
high_stability = [name for name, analysis in stability_analysis.items()
if isinstance(analysis, dict) and analysis.get('stability_rating') == 'high']
report_lines.append(f"• High stability metrics: {', '.join(high_stability) if high_stability else 'None'}")
report_lines.append("")
# Discussion
report_lines.append("DISCUSSION")
report_lines.append("-" * 20)
report_lines.append("The improved HBO algorithm demonstrated effective optimization of CNN")
report_lines.append("hyperparameters for medical image classification. Key findings include:")
report_lines.append("")
# Highlight key achievements
if hbo_improvement > 0.05:
report_lines.append(f"• Significant performance improvement ({hbo_improvement:.1%}) over baseline methods")
if clinical_grade in ['Excellent', 'Good']:
report_lines.append(f"• Achieved clinical-grade performance ({clinical_grade})")
# Add limitations
report_lines.append("")
report_lines.append("Limitations:")
report_lines.append("• Validation performed on single dataset")
report_lines.append("• Simulation-based optimization for demonstration purposes")
report_lines.append("• Requires validation on larger, multi-institutional datasets")
report_lines.append("")
# Conclusions
report_lines.append("CONCLUSIONS")
report_lines.append("-" * 20)
if clinical_grade in ['Excellent', 'Good']:
report_lines.append("The improved HBO algorithm shows strong potential for optimizing medical")
report_lines.append("image classification systems. The achieved performance metrics suggest")
report_lines.append("readiness for clinical validation studies.")
else:
report_lines.append("While the improved HBO algorithm shows promise, additional optimization")
report_lines.append("and validation are required before clinical deployment.")
report_lines.append("")
report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
return "\n".join(report_lines)
def generate_html_dashboard(self, results: Dict[str, Any], plots_dir: Optional[Path] = None) -> str:
"""
Generate interactive HTML dashboard
Args:
results: Complete pipeline results
plots_dir: Directory containing plots
Returns:
HTML dashboard content
"""
if not JINJA2_AVAILABLE:
self.logger.warning("Jinja2 not available. Generating simple HTML dashboard.")
return self._generate_simple_html_dashboard(results)
# HTML template for dashboard
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Image Classification Dashboard</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
.header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; text-align: center; }
.container { max-width: 1200px; margin: 0 auto; }
.card { background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
.metric-box { background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }
.metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
.metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
.grade-excellent { color: #27ae60; }
.grade-good { color: #f39c12; }
.grade-acceptable { color: #e67e22; }
.grade-poor { color: #e74c3c; }
.status-pass { color: #27ae60; font-weight: bold; }
.status-fail { color: #e74c3c; font-weight: bold; }
.recommendations { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; }
.plot-container { text-align: center; margin: 20px 0; }
.plot-container img { max-width: 100%; height: auto; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; margin: 15px 0; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
th { background-color: #34495e; color: white; }
.footer { text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 12px; }
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Medical Image Classification Dashboard</h1>
<p>Comprehensive Analysis Report</p>
<p>Generated on {{ timestamp }}</p>
</div>
<div class="card">
<h2>Executive Summary</h2>
<div class="metrics-grid">
<div class="metric-box">
<div class="metric-value grade-{{ clinical_grade.lower() }}">{{ clinical_grade }}</div>
<div class="metric-label">Clinical Grade</div>
</div>
<div class="metric-box">
<div class="metric-value">{{ "%.1f%%"|format(accuracy * 100) }}</div>
<div class="metric-label">Overall Accuracy</div>
</div>
<div class="metric-box">
<div class="metric-value">{{ "%.1f%%"|format(sensitivity * 100) }}</div>
<div class="metric-label">Sensitivity</div>
</div>
<div class="metric-box">
<div class="metric-value">{{ "%.1f%%"|format(specificity * 100) }}</div>
<div class="metric-label">Specificity</div>
</div>
</div>
</div>
<div class="card">
<h2>Clinical Performance Metrics</h2>
<table>
<thead>
<tr>
<th>Metric</th>
<th>Value</th>
<th>95% CI</th>
<th>Clinical Threshold</th>
<th>Status</th>
</tr>
</thead>
<tbody>
{% for metric in clinical_metrics %}
<tr>
<td>{{ metric.name }}</td>
<td>{{ "%.3f"|format(metric.value) }}</td>
<td>[{{ "%.3f"|format(metric.ci_lower) }}, {{ "%.3f"|format(metric.ci_upper) }}]</td>
<td>{{ "%.3f"|format(metric.threshold) }}</td>
<td class="{{ 'status-pass' if metric.passes else 'status-fail' }}">
{{ 'PASS' if metric.passes else 'FAIL' }}
</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>
{% if recommendations %}
<div class="card">
<h2>Clinical Recommendations</h2>
<div class="recommendations">
<h4>Primary Recommendation:</h4>
<p>{{ primary_recommendation }}</p>
<h4>Detailed Recommendations:</h4>
<ul>
{% for rec in recommendations %}
<li>{{ rec }}</li>
{% endfor %}
</ul>
</div>
</div>
{% endif %}
{% if plots %}
<div class="card">
<h2>Performance Visualizations</h2>
{% for plot in plots %}
<div class="plot-container">
<h4>{{ plot.title }}</h4>
<img src="{{ plot.path }}" alt="{{ plot.title }}">
</div>
{% endfor %}
</div>
{% endif %}
<div class="footer">
<p>This report is generated for research purposes only and does not constitute medical advice.</p>
<p>Medical Image Classification Framework v1.0.0</p>
</div>
</div>
</body>
</html>
"""
# Extract data for template
evaluation_results = results.get('evaluation', {})
validation_results = results.get('validation', {})
clinical_assessment = evaluation_results.get('clinical_assessment', {})
clinical_metrics = evaluation_results.get('clinical_metrics', {})
mean_metrics = validation_results.get('mean_metrics', {})
confidence_intervals = validation_results.get('confidence_intervals', {})
# Prepare template data
template_data = {
'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
'clinical_grade': clinical_assessment.get('grade', 'Unknown'),
'accuracy': mean_metrics.get('accuracy', 0),
'sensitivity': mean_metrics.get('sensitivity', mean_metrics.get('recall', 0)),
'specificity': mean_metrics.get('specificity', 0),
'primary_recommendation': clinical_assessment.get('primary_recommendation', ''),
'recommendations': clinical_assessment.get('detailed_recommendations', [])
}
# Prepare clinical metrics table
clinical_metrics_table = []
for metric_name, metric_data in clinical_metrics.items():
mean_val = metric_data.get('mean', 0)
threshold = metric_data.get('clinical_threshold', 0)
ci = confidence_intervals.get(metric_name, (mean_val, mean_val))
clinical_metrics_table.append({
'name': metric_name.replace('_', ' ').title(),
'value': mean_val,
'ci_lower': ci[0],
'ci_upper': ci[1],
'threshold': threshold,
'passes': mean_val >= threshold
})
template_data['clinical_metrics'] = clinical_metrics_table
# Prepare plots information
plots_info = []
if plots_dir and plots_dir.exists():
plot_files = list(plots_dir.glob('*.png'))
for plot_file in plot_files:
plots_info.append({
'title': plot_file.stem.replace('_', ' ').title(),
'path': str(plot_file.relative_to(self.output_dir.parent))
})
template_data['plots'] = plots_info
# Render template
template = Template(html_template)
html_content = template.render(**template_data)
return html_content
def _generate_simple_html_dashboard(self, results: Dict[str, Any]) -> str:
"""Generate simple HTML dashboard without Jinja2"""
evaluation_results = results.get('evaluation', {})
clinical_assessment = evaluation_results.get('clinical_assessment', {})
html = f"""
<!DOCTYPE html>
<html>
<head>
<title>Medical Classification Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
.card {{ background: white; margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
</style>
</head>
<body>
<div class="header">
<h1>Medical Image Classification Dashboard</h1>
<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
<div class="card">
<h2>Clinical Grade: {clinical_assessment.get('grade', 'Unknown')}</h2>
<p>Recommendation: {clinical_assessment.get('primary_recommendation', 'See detailed report')}</p>
</div>
</body>
</html>
"""
return html
def generate_comprehensive_report(self, results: Dict[str, Any], plots_dir: Optional[Path] = None) -> Dict[str, Any]:
"""
Generate comprehensive report package
Args:
results: Complete pipeline results
plots_dir: Directory containing plots
Returns:
Dictionary with generated report paths and summary
"""
self.logger.info(" Generating comprehensive report package")
generated_reports = {}
# Generate executive summary
if self.config.get('executive_summary', True):
executive_summary = self.generate_executive_summary(results)
exec_path = self.output_dir / 'executive_summary.txt'
with open(exec_path, 'w') as f:
f.write(executive_summary)
generated_reports['executive_summary'] = str(exec_path)
self.logger.info(" Executive summary generated")
# Generate clinical report
if self.clinical_assessment:
clinical_report = self.generate_clinical_report(results)
clinical_path = self.output_dir / 'clinical_report.txt'
with open(clinical_path, 'w') as f:
f.write(clinical_report)
generated_reports['clinical_report'] = str(clinical_path)
self.logger.info(" Clinical report generated")
# Generate technical report
if self.detailed_report:
technical_report = self.generate_technical_report(results)
technical_path = self.output_dir / 'technical_report.txt'
with open(technical_path, 'w') as f:
f.write(technical_report)
generated_reports['technical_report'] = str(technical_path)
self.logger.info(" Technical report generated")
# Generate HTML dashboard
if self.summary_dashboard:
html_dashboard = self.generate_html_dashboard(results, plots_dir)
dashboard_path = self.output_dir / 'dashboard.html'
with open(dashboard_path, 'w') as f:
f.write(html_dashboard)
generated_reports['html_dashboard'] = str(dashboard_path)
self.logger.info(" HTML dashboard generated")
# Generate complete results JSON
complete_results_path = self.output_dir / 'complete_results.json'
with open(complete_results_path, 'w') as f:
json.dump(results, f, indent=2, default=str)
generated_reports['complete_results'] = str(complete_results_path)
# Create clinical summary
evaluation_results = results.get('evaluation', {})
clinical_assessment = evaluation_results.get('clinical_assessment', {})
clinical_summary = {
'clinical_grade': clinical_assessment.get('grade', 'Unknown'),
'clinical_readiness': clinical_assessment.get('clinical_readiness', 'unknown'),
'primary_recommendation': clinical_assessment.get('primary_recommendation', ''),
'risk_level': clinical_assessment.get('risk_assessment', {}).get('overall_risk_level', 'unknown'),
'reports_generated': len(generated_reports),
'generation_timestamp': datetime.now().isoformat()
}
# Save report generation summary
summary_path = self.output_dir / 'report_generation_summary.json'
with open(summary_path, 'w') as f:
json.dump({
'generated_reports': generated_reports,
'clinical_summary': clinical_summary,
'config': self.config
}, f, indent=2)
self.logger.info(f"Comprehensive report package generated ({len(generated_reports)} reports)")
self.logger.info(f" Clinical grade: {clinical_summary['clinical_grade']}")
return {
'generated_reports': generated_reports,
'dashboard_path': generated_reports.get('html_dashboard'),
'clinical_summary': clinical_summary
}
if __name__ == "__main__":
# Example usage
config = {
'generate_plots': True,
'clinical_assessment': True,
'detailed_report': True,
'summary_dashboard': True,
'executive_summary': True
}
reporter = ReportGenerator(config, './reporting_output')
print(" Report Generator ready for use!")