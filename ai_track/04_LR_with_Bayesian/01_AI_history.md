## 1. Historical Perspective on AI

1. (Knowledge Base) -> (Inference Engine) -> Conclusion
                          ^observations
    
    1. Knowledge base : encodes what we know about the world. The KB is often called a model, giving rise to the term **model-based reasoning**.
    2. Reasoner(Inference engine) acts on the knowledge base to answer queries of interest.

    => 
    (Statement in Logic) -> (Logical Deduction) -> Conclusion
                          ^observations

    1. Express knowledge base using statements in a suitable logic.
    
    2. Use logical deduction in realizing the reasoning engine.
  *- Later these statements were revised(deduction has limitations)*
       
    
2. The Limits of Deduction
    Deductive logic is not capable of dealing with assumptions that tend to be prevalent in commonsense reasoning.
    
    
    "If a bird is normal, it will fly." (stated in 1958)
    The common belief that a bird would fly cannot be logically deduced from the above fact unless we further assume that the bird we just saw is normal.
    => **Hence, the belief in a flying bird is the result of a logical deduction applied to a mixture of facts and assumptions.**
    This abililty to dynamically assert and retract assumptions depending on what is currently known is quite typical in commonsense reasoning. = outside the realm of deductive logic.
    **Deductive logic is monotonic:** once we deduce something from a knowledge base(the bird flies), we can never invalidate the deduction by acquiring more knowledge(the bird has a broken wing).
    
3. Monotonicity
	- if A(set of premises) logically implies b, then A and C(additional premises) will also logically imply b.
	- Hence, no deductive logic is capable of producing the reasoning process described above with regards to flying birds.

4. The qualification problem
	1. "Birds fly", fall into an inconsistency if it encounters a bird that cannot fly.
	2. "If a bird is normal, it flies", it would not know whether the bird is normal or not- contrary to what most humans will do.(stated in 1970s) 
	The failture of deductive logic => counteracting this failure
	
5. Non-monotonic logics
	1. Equip classical logic with an ability to jump into certain conclusions
	2. Install the notion of assumption
	3. Allow one to dynamically assert and retract depending on what else i known.
	=> Regulating conflicts is really difficult.
	=> **Alternative solution : degree of belief**
	
## 2. From logic to probability
1. Degree of belief : a number that one assigns to a proposition.
	We assign the degree of belief to the bird's normality 99%, then use it to derive a corresponding degree of belief in the bird's flying ability.

2. Degree of belief + non-monotonicity
	degree of belief revisable, depending on what else is known.
	ex) bird's normality to 20% after learning new observations.

3. Bayesian Networks
	= modeling tool(representation device) to organize one's knowledge about a particular situation into a coherent whole.
	1. consistent and complete
	2. modular
	3. compact (can specify an exponentially-sized probabililty distribution using a polynomial number of probabilities)

## 3. From models to functions
1. Model-based approach : represent & reason | logic & probability
2. Model-blind approch : fit functions to data, Neural Networks
	Issue: size


	


