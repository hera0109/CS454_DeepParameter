import org.moeaframework.Executor;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.problem.AbstractProblem;

public class DeepParameterTuning {

	public static class DeepParameterTuningProblem extends AbstractProblem {
	
		public static final int[] genotype = {0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,2,2,2,1,0,2,2,0,0,0,0,0,2,2,1,1,1,1,0,0,1,1,1,2,31,0,0,1,1,0,0,0,0};	
		private static final int numObjectives=2;
		public DeepParameterTuningProblem() {
			super(genotype.length,numObjectives);
		}

		public Solution newSolution() {
			Solution solution = new Solution(getNumberOfVariables(),getNumberOfObjectives());
			for(int i=0; i < getNumberOfVariables(); i++) {
				solution.setVariable(i, new BinaryIntegerVariable(genotype[i], -64, 64));
			}

			return solution;
		}

		public void evaluate(Solution solution) {
			double[] objectiveResults= new double[numObjectives];
		
			int[] values = getValues(solution);
		
			//Just for testing purposes

			int total = 0;
			for(int i=0; i < values.length; i++){
				total+=values[i];
			}
			objectiveResults[0]=total;
			objectiveResults[1]=total*-1;

			solution.setObjectives(objectiveResults);
		}

		private static int[] getValues(Solution solution) {
			int [] toReturn = new int [solution.getNumberOfVariables()];
			for(int i=0; i < solution.getNumberOfVariables(); i++){
				toReturn[i] = ((BinaryIntegerVariable)solution.getVariable(i)).getValue();
			}
			return toReturn;
		}
	}

	public static void main(String[] args) {
		NondominatedPopulation result = new Executor()
			.withProblemClass(DeepParameterTuningProblem.class)
			.withAlgorithm("NSGAII")
			.withMaxEvaluations(1000000)
			.run();
		
		//Display the results
		for(Solution solution : result) {
			System.out.format("%.4f	%.4f%n",
				solution.getObjective(0),
				solution.getObjective(1));
		}
		
	}
}
