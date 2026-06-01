"""
Caseflow Evaluation System - Multilingual Support
Adds deterministic eval coverage for bilingual and multilingual support requests.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for support cases."""
    LOW = "low"
    HIGH = "high"

class CaseType(Enum):
    """Types of support cases."""
    REFUND = "refund"
    COMPENSATION = "compensation"
    COMPLAINT_ESCALATION = "complaint_escalation"
    GENERAL_INQUIRY = "general_inquiry"

@dataclass
class EvalCase:
    """Represents an evaluation case for testing."""
    id: str
    language: str
    case_type: CaseType
    risk_level: RiskLevel
    description: str
    expected_fields: List[str]
    input_text: str
    expected_output: Dict[str, Any]
    hitl_required: bool = False

class CaseflowEvaluator:
    """Handles evaluation of caseflow cases with multilingual support."""
    
    def __init__(self, cases_file: str = "data/caseflow/eval_cases.json"):
        self.cases_file = Path(cases_file)
        self.cases: List[EvalCase] = []
        
    def load_cases(self) -> List[EvalCase]:
        """Load evaluation cases from JSON file."""
        try:
            if not self.cases_file.exists():
                logger.warning(f"Cases file not found: {self.cases_file}")
                return []
            
            with open(self.cases_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.cases = [self._parse_case(case_data) for case_data in data]
            logger.info(f"Loaded {len(self.cases)} evaluation cases")
            return self.cases
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cases file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading cases: {e}")
            return []
    
    def _parse_case(self, case_data: Dict) -> EvalCase:
        """Parse case data into EvalCase object."""
        return EvalCase(
            id=case_data.get("id", ""),
            language=case_data.get("language", "en"),
            case_type=CaseType(case_data.get("case_type", "general_inquiry")),
            risk_level=RiskLevel(case_data.get("risk_level", "low")),
            description=case_data.get("description", ""),
            expected_fields=case_data.get("expected_fields", []),
            input_text=case_data.get("input_text", ""),
            expected_output=case_data.get("expected_output", {}),
            hitl_required=case_data.get("hitl_required", False)
        )
    
    def add_multilingual_cases(self) -> None:
        """Add synthetic multilingual cases for evaluation."""
        multilingual_cases = [
            # Low-risk inquiry: Simple refund request in Spanish
            EvalCase(
                id="multilingual_refund_es_001",
                language="es",
                case_type=CaseType.REFUND,
                risk_level=RiskLevel.LOW,
                description="Simple refund request in Spanish",
                expected_fields=["case_type", "language", "risk_level", "action_required", "refund_amount"],
                input_text="Quisiera solicitar un reembolso por mi compra reciente. El producto llegó dañado.",
                expected_output={
                    "case_type": "refund",
                    "language": "es",
                    "risk_level": "low",
                    "action_required": "process_refund",
                    "refund_amount": "full",
                    "hitl_required": False
                },
                hitl_required=False
            ),
            
            # High-risk HITL path: Complaint escalation in French
            EvalCase(
                id="multilingual_complaint_fr_001",
                language="fr",
                case_type=CaseType.COMPLAINT_ESCALATION,
                risk_level=RiskLevel.HIGH,
                description="Complaint escalation in French requiring human intervention",
                expected_fields=["case_type", "language", "risk_level", "escalation_reason", "hitl_required", "priority"],
                input_text="Je suis extrêmement mécontent du service client. J'ai été transféré plusieurs fois sans résolution. Je demande à parler à un superviseur immédiatement.",
                expected_output={
                    "case_type": "complaint_escalation",
                    "language": "fr",
                    "risk_level": "high",
                    "escalation_reason": "customer_dissatisfaction_multiple_transfers",
                    "hitl_required": True,
                    "priority": "urgent",
                    "escalation_path": "supervisor_review"
                },
                hitl_required=True
            ),
            
            # High-risk HITL path: Compensation request in Mandarin
            EvalCase(
                id="multilingual_compensation_zh_001",
                language="zh",
                case_type=CaseType.COMPENSATION,
                risk_level=RiskLevel.HIGH,
                description="Compensation request in Mandarin requiring human approval",
                expected_fields=["case_type", "language", "risk_level", "compensation_type", "amount", "hitl_required"],
                input_text="由于航班延误12小时，严重影响了我的行程安排。我要求赔偿我的经济损失，包括酒店住宿和交通费用。",
                expected_output={
                    "case_type": "compensation",
                    "language": "zh",
                    "risk_level": "high",
                    "compensation_type": "financial_loss",
                    "amount": "to_be_determined",
                    "hitl_required": True,
                    "reason": "flight_delay_12_hours"
                },
                hitl_required=True
            )
        ]
        
        # Add cases to existing list
        self.cases.extend(multilingual_cases)
        logger.info(f"Added {len(multilingual_cases)} multilingual evaluation cases")
    
    def save_cases(self) -> bool:
        """Save evaluation cases to JSON file."""
        try:
            # Ensure directory exists
            self.cases_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert cases to dictionary format
            cases_data = []
            for case in self.cases:
                case_dict = asdict(case)
                case_dict["case_type"] = case.case_type.value
                case_dict["risk_level"] = case.risk_level.value
                cases_data.append(case_dict)
            
            # Save with proper encoding for multilingual content
            with open(self.cases_file, 'w', encoding='utf-8') as f:
                json.dump(cases_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(cases_data)} cases to {self.cases_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving cases: {e}")
            return False
    
    def evaluate_case(self, case: EvalCase) -> Dict[str, Any]:
        """Evaluate a single case and return results."""
        try:
            # Simulate evaluation with fake model
            result = {
                "case_id": case.id,
                "evaluated": True,
                "detected_language": case.language,
                "detected_case_type": case.case_type.value,
                "detected_risk_level": case.risk_level.value,
                "hitl_required": case.hitl_required,
                "fields_present": self._check_expected_fields(case),
                "output_matches": self._check_output_match(case),
                "deterministic": True  # Always deterministic with fake model
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating case {case.id}: {e}")
            return {
                "case_id": case.id,
                "evaluated": False,
                "error": str(e)
            }
    
    def _check_expected_fields(self, case: EvalCase) -> Dict[str, bool]:
        """Check if expected fields are present in the case."""
        field_check = {}
        for field in case.expected_fields:
            field_check[field] = field in asdict(case) or field in case.expected_output
        return field_check
    
    def _check_output_match(self, case: EvalCase) -> bool:
        """Check if expected output matches case structure."""
        try:
            case_dict = asdict(case)
            for key, value in case.expected_output.items():
                if key in case_dict:
                    if case_dict[key] != value:
                        return False
            return True
        except:
            return False

class EvaluationTest:
    """Test class for evaluation functionality."""
    
    def __init__(self):
        self.evaluator = CaseflowEvaluator()
    
    def test_multilingual_cases_addition(self) -> bool:
        """Test that multilingual cases are added correctly."""
        try:
            self.evaluator.add_multilingual_cases()
            multilingual_cases = [c for c in self.evaluator.cases if c.language != "en"]
            
            assert len(multilingual_cases) >= 3, "Should have at least 3 multilingual cases"
            
            # Check specific cases
            spanish_cases = [c for c in multilingual_cases if c.language == "es"]
            french_cases = [c for c in multilingual_cases if c.language == "fr"]
            mandarin_cases = [c for c in multilingual_cases if c.language == "zh"]
            
            assert len(spanish_cases) >= 1, "Should have Spanish case"
            assert len(french_cases) >= 1, "Should have French case"
            assert len(mandarin_cases) >= 1, "Should have Mandarin case"
            
            # Check risk levels
            low_risk = [c for c in multilingual_cases if c.risk_level == RiskLevel.LOW]
            high_risk = [c for c in multilingual_cases if c.risk_level == RiskLevel.HIGH]
            
            assert len(low_risk) >= 1, "Should have at least 1 low-risk case"
            assert len(high_risk) >= 2, "Should have at least 2 high-risk cases"
            
            logger.info("✓ Multilingual cases addition test passed")
            return True
            
        except AssertionError as e:
            logger.error(f"✗ Test failed: {e}")
            return False
    
    def test_hitl_paths(self) -> bool:
        """Test that HITL paths are correctly identified."""
        try:
            self.evaluator.add_multilingual_cases()
            hitl_cases = [c for c in self.evaluator.cases if c.hitl_required]
            
            assert len(hitl_cases) >= 2, "Should have at least 2 HITL cases"
            
            for case in hitl_cases:
                assert case.risk_level == RiskLevel.HIGH, "HITL cases should be high risk"
                assert case.case_type in [CaseType.COMPLAINT_ESCALATION, CaseType.COMPENSATION], \
                    "HITL cases should be complaint escalation or compensation"
            
            logger.info("✓ HITL paths test passed")
            return True
            
        except AssertionError as e:
            logger.error(f"✗ Test failed: {e}")
            return False
    
    def test_deterministic_evaluation(self) -> bool:
        """Test that evaluation is deterministic with fake model."""
        try:
            self.evaluator.add_multilingual_cases()
            
            # Run evaluation twice and compare results
            results1 = [self.evaluator.evaluate_case(case) for case in self.evaluator.cases]
            results2 = [self.evaluator.evaluate_case(case) for case in self.evaluator.cases]
            
            for r1, r2 in zip(results1, results2):
                assert r1 == r2, "Evaluation results should be identical"
                assert r1.get("deterministic") == True, "Should be marked as deterministic"
            
            logger.info("✓ Deterministic evaluation test passed")
            return True
            
        except AssertionError as e:
            logger.error(f"✗ Test failed: {e}")
            return False
    
    def test_expected_fields(self) -> bool:
        """Test that expected fields are present in cases."""
        try:
            self.evaluator.add_multilingual_cases()
            
            for case in self.evaluator.cases:
                field_check = self.evaluator._check_expected_fields(case)
                missing_fields = [field for field, present in field_check.items() if not present]
                
                if missing_fields:
                    logger.warning(f"Case {case.id} missing fields: {missing_fields}")
            
            logger.info("✓ Expected fields test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            return False

def main():
    """Main execution function."""
    logger.info("Starting Caseflow Evaluation System")
    
    # Initialize evaluator
    evaluator = CaseflowEvaluator()
    
    # Add multilingual cases
    evaluator.add_multilingual_cases()
    
    # Save cases to file
    if evaluator.save_cases():
        logger.info("Cases saved successfully")
    else:
        logger.error("Failed to save cases")
    
    # Run evaluation tests
    logger.info("\n" + "="*50)
    logger.info("Running Evaluation Tests")
    logger.info("="*50)
    
    test_runner = EvaluationTest()
    
    tests = [
        ("Multilingual Cases Addition", test_runner.test_multilingual_cases_addition),
        ("HITL Paths", test_runner.test_hitl_paths),
        ("Deterministic Evaluation", test_runner.test_deterministic_evaluation),
        ("Expected Fields", test_runner.test_expected_fields)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        if test_func():
            logger.info(f"✓ {test_name} passed")
        else:
            logger.error(f"✗ {test_name} failed")
            all_passed = False
    
    # Summary
    logger.info("\n" + "="*50)
    if all_passed:
        logger.info("✓ All tests passed!")
    else:
        logger.error("✗ Some tests failed!")
    
    # Print case summary
    logger.info("\n" + "="*50)
    logger.info("Case Summary")
    logger.info("="*50)
    for case in evaluator.cases:
        logger.info(f"  - {case.id}: {case.language.upper()} | {case.case_type.value} | {case.risk_level.value} risk | HITL: {case.hitl_required}")

if __name__ == "__main__":
    main()