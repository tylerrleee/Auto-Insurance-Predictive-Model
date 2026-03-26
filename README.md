# 



Predictive model that estimates the severity of insurance claims of XXX 


# Goals

1. Cross-validate regression/emsemble models
2. Feature engineering, encoding, handling missing values
3. Document trade-offs between model complexity vs. interpretability
4. Summary deck w/ findings
5. Analyze feature importance and recommend what data matters most
6. Handling diverse feature scope (multi-columns)

# Scenarios


- **False Negative (missed claim)**: Risky policyholder charged standard rate → Company loses money on claims
- **False Positive (wrongly flagged)**: Safe policyholder charged higher premium → Might lose customer to competitor

Let's assume:
- Average claim cost: **\$5,000**
- Premium increase for flagged customers: **\$500/year**
- Customer churn rate if wrongly flagged: **20%**
- Lifetime value of lost customer: **\$2,000**