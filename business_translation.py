"""
Business Translation Framework
Converting philosophical concepts into industry-standard terminology while preserving core intent
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

class AudienceType(Enum):
    BUSINESS = "business"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    DEVELOPER = "developer"
    INVESTOR = "investor"
    EXECUTIVE = "executive"

@dataclass
class TranslationMapping:
    original: str
    business: str
    technical: str
    academic: str
    context: Optional[str] = None

class BusinessTranslationFramework:
    """
    Core framework for translating philosophical/spiritual concepts 
    into industry-standard business terminology
    """
    
    def __init__(self):
        self.translation_dictionary = self._build_translation_dictionary()
        self.audience_strategies = self._build_audience_strategies()
        
    def _build_translation_dictionary(self) -> Dict[str, TranslationMapping]:
        """Build the master translation dictionary"""
        mappings = [
            # Foundational Concepts
            TranslationMapping(
                "Sacred Architecture",
                "Sustainable Platform Architecture",
                "User-Centered System Design", 
                "Evidence-Based Technology Framework"
            ),
            TranslationMapping(
                "Digital Mercy",
                "Cognitive Load Optimization",
                "Adaptive User Interface Design",
                "Human-Computer Interaction Best Practices"
            ),
            TranslationMapping(
                "Kindness Algorithm",
                "User-Centered Algorithm",
                "Value-Alignment Framework",
                "Behavioral Psychology Computing"
            ),
            TranslationMapping(
                "Trust Layer",
                "Transparency Protocol",
                "Algorithmic Accountability Framework",
                "Ethical Computing Standard"
            ),
            TranslationMapping(
                "Sacred Promise",
                "Platform Commitment",
                "Service Level Agreement",
                "User Rights Declaration"
            ),
            TranslationMapping(
                "Gentle Power",
                "Sustainable Influence",
                "User Empowerment Design",
                "Participatory Technology"
            ),
            TranslationMapping(
                "Building with Love",
                "Evidence-Based Development",
                "User-Centered Engineering",
                "Human-Factors Development"
            ),
            TranslationMapping(
                "Consciousness",
                "User Experience",
                "Cognitive Performance",
                "Human Factors"
            ),
            TranslationMapping(
                "Flourishing",
                "Measurable Outcomes",
                "Performance Optimization",
                "User Goal Achievement"
            ),
            TranslationMapping(
                "All Our Relations",
                "Stakeholder Value",
                "Network Effects",
                "Ecosystem Impact"
            ),
            
            # Process and Methodology Terms
            TranslationMapping(
                "The Builder's Oath",
                "Developer Best Practices Framework",
                "Ethical Engineering Standards",
                "Professional Development Guidelines"
            ),
            TranslationMapping(
                "Sacred Side Projects",
                "Innovation Lab Projects",
                "R&D Portfolio",
                "Experimental Development"
            ),
            TranslationMapping(
                "Inner Development",
                "Professional Development",
                "Skills Enhancement",
                "Competency Building"
            ),
            TranslationMapping(
                "Moral Injury",
                "Role-Values Misalignment",
                "Job Satisfaction Issues",
                "Professional Burnout"
            ),
            TranslationMapping(
                "Spiritual Practice",
                "Mindfulness Training",
                "Focus Enhancement",
                "Cognitive Training"
            ),
            TranslationMapping(
                "Community Healing",
                "User Community Building",
                "Network Development",
                "Social Capital Formation"
            ),
            
            # Technical Implementation Terms
            TranslationMapping(
                "Attention Restoration",
                "Focus Optimization",
                "Cognitive Load Management",
                "Attention Research Application"
            ),
            TranslationMapping(
                "Consensual Computing",
                "Permission-Based Architecture",
                "Explicit Consent Framework",
                "Privacy-Preserving Computing"
            ),
            TranslationMapping(
                "Algorithmic Transparency",
                "Explainable AI",
                "Interpretable Machine Learning",
                "Algorithmic Accountability"
            ),
            TranslationMapping(
                "Data Sovereignty",
                "User Data Ownership",
                "Personal Data Management",
                "Information Self-Determination"
            ),
            TranslationMapping(
                "Gentle Notifications",
                "Adaptive Notification System",
                "Context-Aware Interruption",
                "Respectful Computing"
            ),
            TranslationMapping(
                "Presence-Centered Design",
                "Focus-Optimized Interface",
                "Distraction-Minimized UX",
                "Attention-Aware Design"
            ),
            
            # Business Model Terms
            TranslationMapping(
                "Gift Economy",
                "Community-Supported Business",
                "Cooperative Platform Model",
                "Alternative Economic Framework"
            ),
            TranslationMapping(
                "Regenerative Economics",
                "Sustainable Business Model",
                "Circular Value Creation",
                "Long-term Value Optimization"
            ),
            TranslationMapping(
                "The Commons",
                "Shared Resource Platform",
                "Open Source Ecosystem",
                "Public Goods Infrastructure"
            ),
            TranslationMapping(
                "Mutual Aid",
                "Peer-to-Peer Support",
                "Distributed Assistance Network",
                "Community Resilience System"
            )
        ]
        
        return {mapping.original.lower(): mapping for mapping in mappings}
    
    def _build_audience_strategies(self) -> Dict[AudienceType, Dict[str, str]]:
        """Build audience-specific translation strategies"""
        return {
            AudienceType.DEVELOPER: {
                "focus": "engineering terminology, APIs, technical specifications",
                "emphasize": "performance metrics, scalability, reliability",
                "avoid": "spiritual/emotional language",
                "include": "implementation details, code examples"
            },
            AudienceType.BUSINESS: {
                "focus": "ROI, competitive advantage, market opportunity", 
                "emphasize": "case studies, data-driven evidence",
                "avoid": "idealistic language without business rationale",
                "include": "risk mitigation, future-proofing strategy"
            },
            AudienceType.ACADEMIC: {
                "focus": "established research, peer-reviewed citations",
                "emphasize": "statistical evidence, testable hypotheses",
                "avoid": "unsupported claims",
                "include": "empirical research framework"
            },
            AudienceType.INVESTOR: {
                "focus": "market size, growth potential, competitive moats",
                "emphasize": "regulatory alignment, risk mitigation",
                "avoid": "emotional appeals without data",
                "include": "path to profitability and scale"
            },
            AudienceType.EXECUTIVE: {
                "focus": "strategic advantage, business impact",
                "emphasize": "competitive positioning, market opportunity",
                "avoid": "technical details without business context",
                "include": "leadership implications, decision frameworks"
            }
        }
    
    def translate_term(self, original_term: str, audience: AudienceType) -> Optional[str]:
        """
        Translate a single term for a specific audience
        
        Args:
            original_term: The philosophical/spiritual term to translate
            audience: Target audience type
            
        Returns:
            Translated term or None if not found
        """
        mapping = self.translation_dictionary.get(original_term.lower())
        if not mapping:
            return None
            
        if audience == AudienceType.BUSINESS:
            return mapping.business
        elif audience == AudienceType.TECHNICAL:
            return mapping.technical
        elif audience == AudienceType.ACADEMIC:
            return mapping.academic
        elif audience == AudienceType.DEVELOPER:
            return mapping.technical  # Use technical for developers
        elif audience in [AudienceType.INVESTOR, AudienceType.EXECUTIVE]:
            return mapping.business  # Use business for executives/investors
        
        return mapping.business  # Default to business translation
    
    def translate_text(self, text: str, audience: AudienceType) -> str:
        """
        Translate a full text by replacing known terms
        
        Args:
            text: Text containing philosophical/spiritual terminology
            audience: Target audience type
            
        Returns:
            Text with terms translated for the target audience
        """
        translated_text = text
        
        # Sort by length (longest first) to avoid partial replacements
        terms = sorted(self.translation_dictionary.keys(), key=len, reverse=True)
        
        for original_term in terms:
            if original_term in translated_text.lower():
                translated_term = self.translate_term(original_term, audience)
                if translated_term:
                    # Case-insensitive replacement while preserving surrounding case
                    import re
                    pattern = re.compile(re.escape(original_term), re.IGNORECASE)
                    translated_text = pattern.sub(translated_term, translated_text)
        
        return translated_text
    
    def get_context_examples(self, audience: AudienceType) -> List[Dict[str, str]]:
        """
        Get context-specific translation examples for an audience
        
        Args:
            audience: Target audience type
            
        Returns:
            List of example translations with original and translated versions
        """
        examples = {
            AudienceType.DEVELOPER: [
                {
                    "original": "Build with love and consciousness",
                    "translation": "Implement user-centered design with cognitive science principles"
                },
                {
                    "original": "Sacred responsibility of code", 
                    "translation": "Professional accountability in software development"
                },
                {
                    "original": "Technology that serves consciousness",
                    "translation": "Software that optimizes for user cognitive performance"
                }
            ],
            AudienceType.BUSINESS: [
                {
                    "original": "Love scales better than extraction",
                    "translation": "User-centered platforms achieve higher customer lifetime value than exploitation-based models"
                },
                {
                    "original": "Sacred Architecture creates flourishing",
                    "translation": "Sustainable platform architecture delivers measurable ROI through user satisfaction"
                },
                {
                    "original": "Digital mercy for vulnerable users",
                    "translation": "Accessibility-first design reducing user friction and support costs"
                }
            ],
            AudienceType.INVESTOR: [
                {
                    "original": "Building technology that heals",
                    "translation": "Developing platforms with positive user outcomes and sustainable growth metrics"
                },
                {
                    "original": "Trust as infrastructure", 
                    "translation": "Transparency and user control as competitive advantages reducing regulatory risk"
                },
                {
                    "original": "Serving human flourishing",
                    "translation": "Optimizing for long-term user value and ecosystem health"
                }
            ],
            AudienceType.ACADEMIC: [
                {
                    "original": "Sacred side projects",
                    "translation": "Experimental development in human-computer interaction"
                },
                {
                    "original": "Spiritual dimension of technology",
                    "translation": "Ethical considerations in computational system design"
                },
                {
                    "original": "Consciousness and code",
                    "translation": "Cognitive science applications in software engineering"
                }
            ]
        }
        
        return examples.get(audience, [])
    
    def validate_translation(self, original: str, translated: str, audience: AudienceType) -> Dict[str, bool]:
        """
        Validate a translation against framework principles
        
        Args:
            original: Original philosophical text
            translated: Proposed translation
            audience: Target audience
            
        Returns:
            Validation results with specific checks
        """
        validation = {
            "preserves_core_intent": True,  # Would need semantic analysis
            "uses_professional_terminology": True,  # Would need terminology validation
            "audience_appropriate": True,  # Would need audience analysis
            "actionable_elements_clear": True,  # Would need action extraction
            "evidence_based": True,  # Would need fact checking
            "avoids_red_flags": True  # Would need red flag detection
        }
        
        # Simple heuristic checks
        red_flags = ["spiritual", "sacred", "consciousness", "love", "healing", "divine"]
        if audience in [AudienceType.BUSINESS, AudienceType.INVESTOR, AudienceType.EXECUTIVE]:
            for flag in red_flags:
                if flag.lower() in translated.lower():
                    validation["avoids_red_flags"] = False
                    break
        
        return validation
    
    def get_implementation_guidelines(self) -> Dict[str, List[str]]:
        """Get step-by-step implementation guidelines"""
        return {
            "audience_analysis": [
                "Identify primary audience (developers, business, investors, academic)",
                "Determine their primary concerns and success metrics", 
                "Choose appropriate translation column from dictionary"
            ],
            "core_message_preservation": [
                "Identify the essential concept being communicated",
                "Ensure translation maintains the practical implementation",
                "Verify the actionable element remains clear"
            ],
            "evidence_integration": [
                "Replace emotional appeals with data points",
                "Add specific metrics and performance indicators",
                "Include concrete examples and case studies"
            ],
            "professional_validation": [
                "Use industry-standard terminology",
                "Reference established frameworks and best practices",
                "Align with current technology trends and concerns"
            ]
        }
    
    def get_quality_checklist(self) -> Dict[str, List[str]]:
        """Get quality assurance checklist"""
        return {
            "before_translation": [
                "Core concept clearly identified",
                "Practical implementation defined", 
                "Measurable outcomes specified"
            ],
            "after_translation": [
                "Language appropriate for target audience",
                "Core intent preserved",
                "Actionable elements clear",
                "Evidence-based claims included",
                "Professional terminology used",
                "Industry relevance demonstrated"
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    framework = BusinessTranslationFramework()
    
    # Test single term translation
    print("Term Translation Examples:")
    print(f"Sacred Architecture -> Business: {framework.translate_term('Sacred Architecture', AudienceType.BUSINESS)}")
    print(f"Kindness Algorithm -> Technical: {framework.translate_term('Kindness Algorithm', AudienceType.TECHNICAL)}")
    print(f"Digital Mercy -> Academic: {framework.translate_term('Digital Mercy', AudienceType.ACADEMIC)}")
    
    # Test text translation
    print("\nText Translation Example:")
    original_text = "Our Sacred Architecture implements the Kindness Algorithm to create Digital Mercy for all users through Conscious Computing."
    translated_business = framework.translate_text(original_text, AudienceType.BUSINESS)
    print(f"Original: {original_text}")
    print(f"Business: {translated_business}")
    
    # Test context examples
    print("\nBusiness Context Examples:")
    examples = framework.get_context_examples(AudienceType.BUSINESS)
    for example in examples[:2]:  # Show first 2 examples
        print(f"Original: {example['original']}")
        print(f"Translation: {example['translation']}")
        print()