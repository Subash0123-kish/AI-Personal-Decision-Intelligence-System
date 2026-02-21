"""
AI Recommendation Module
Converts predictions into human-readable, actionable advice using LLM
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AIRecommender:
    """
    Generates human-readable recommendations using LLM APIs
    Converts model predictions, SHAP explanations, and risk levels into actionable advice
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI recommender
        
        Args:
            api_key: OpenAI API key (or from environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: OpenAI package not installed. Install with: pip install openai")
    
    def _create_prompt(self, prediction: float, decision_score: float, 
                      risk_level: str, domain: str,
                      feature_contributions: Optional[Dict[str, float]] = None,
                      feature_values: Optional[Dict[str, Any]] = None) -> str:
        """
        Create prompt for LLM
        
        Args:
            prediction: Model prediction
            decision_score: Decision score (0-1)
            risk_level: Risk level (Low/Medium/High)
            domain: Domain context (finance, career, health, lifestyle)
            feature_contributions: Feature contribution dictionary
            feature_values: Actual feature values
            
        Returns:
            Formatted prompt string
        """
        # Domain-specific context
        domain_contexts = {
            'finance': 'financial decision',
            'career': 'career decision',
            'health': 'health-related decision',
            'lifestyle': 'lifestyle decision'
        }
        
        context = domain_contexts.get(domain.lower(), 'decision')
        
        prompt = f"""You are an expert advisor helping with a {context}.

Prediction Analysis:
- Predicted Outcome: {prediction:.4f}
- Decision Score: {decision_score:.4f} (on a scale of 0-1)
- Risk Level: {risk_level}

"""
        
        if feature_contributions:
            prompt += "Key Factors Influencing This Decision:\n"
            # Get top 5 contributing factors
            top_factors = sorted(feature_contributions.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:5]
            for feature, contribution in top_factors:
                direction = "positively" if contribution > 0 else "negatively"
                value = feature_values.get(feature, "N/A") if feature_values else "N/A"
                prompt += f"- {feature} (current value: {value}): {direction} impacts the decision\n"
            prompt += "\n"
        
        prompt += f"""Based on this analysis, provide:
1. A clear, concise explanation of what this prediction means
2. Actionable recommendations (2-3 specific steps)
3. Risk mitigation strategies if the risk level is Medium or High
4. Next steps the person should consider

Write in a friendly, professional tone. Be specific and practical.
Keep the response under 200 words.

Recommendation:"""

        return prompt
    
    def generate_recommendation(self, prediction: float, decision_score: float,
                               risk_level: str, domain: str = 'general',
                               feature_contributions: Optional[Dict[str, float]] = None,
                               feature_values: Optional[Dict[str, Any]] = None,
                               model: str = 'gpt-3.5-turbo') -> str:
        """
        Generate recommendation using LLM
        
        Args:
            prediction: Model prediction
            decision_score: Decision score (0-1)
            risk_level: Risk level (Low/Medium/High)
            domain: Domain context
            feature_contributions: Feature contribution dictionary
            feature_values: Actual feature values
            model: LLM model to use
            
        Returns:
            Generated recommendation text
        """
        if not self.client:
            # Fallback recommendation without LLM
            return self._generate_fallback_recommendation(
                prediction, decision_score, risk_level, domain
            )
        
        try:
            prompt = self._create_prompt(
                prediction, decision_score, risk_level, domain,
                feature_contributions, feature_values
            )
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful decision intelligence advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            recommendation = response.choices[0].message.content.strip()
            return recommendation
        
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return self._generate_fallback_recommendation(
                prediction, decision_score, risk_level, domain
            )
    
    def _generate_fallback_recommendation(self, prediction: float, 
                                         decision_score: float,
                                         risk_level: str,
                                         domain: str) -> str:
        """
        Generate fallback recommendation without LLM
        
        Args:
            prediction: Model prediction
            decision_score: Decision score
            risk_level: Risk level
            domain: Domain context
            
        Returns:
            Fallback recommendation text
        """
        recommendation_parts = []
        
        recommendation_parts.append(f"Based on your {domain} analysis:\n")
        recommendation_parts.append(f"Decision Score: {decision_score:.2%}\n")
        recommendation_parts.append(f"Risk Level: {risk_level}\n\n")
        
        if risk_level == 'Low':
            recommendation_parts.append("This appears to be a low-risk decision. ")
            recommendation_parts.append("Consider proceeding with caution and monitoring outcomes.\n")
        elif risk_level == 'Medium':
            recommendation_parts.append("This is a medium-risk decision. ")
            recommendation_parts.append("Review all factors carefully and consider seeking additional advice.\n")
        else:  # High
            recommendation_parts.append("This is a high-risk decision. ")
            recommendation_parts.append("Strongly consider consulting with experts and exploring alternatives.\n")
        
        recommendation_parts.append("\nRecommendations:")
        recommendation_parts.append("1. Review all contributing factors carefully")
        recommendation_parts.append("2. Consider alternative options")
        recommendation_parts.append("3. Monitor the situation closely if proceeding")
        
        return "\n".join(recommendation_parts)
    
    def generate_from_explanation(self, explanation_data: Dict[str, Any],
                                  domain: str = 'general') -> str:
        """
        Generate recommendation from explanation data
        
        Args:
            explanation_data: Dictionary with prediction, scores, and explanations
            domain: Domain context
            
        Returns:
            Generated recommendation
        """
        prediction = explanation_data.get('prediction', 0)
        decision_score = explanation_data.get('decision_score', 0)
        risk_level = explanation_data.get('risk_level', 'Medium')
        feature_contributions = explanation_data.get('feature_contributions', {})
        feature_values = explanation_data.get('feature_values', {})
        
        return self.generate_recommendation(
            prediction=prediction,
            decision_score=decision_score,
            risk_level=risk_level,
            domain=domain,
            feature_contributions=feature_contributions,
            feature_values=feature_values
        )
