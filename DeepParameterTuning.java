import org.moeaframework.Executor;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.core.variable.RealVariable;
import org.moeaframework.problem.AbstractProblem;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.File;
import java.util.Arrays;

public class DeepParameterTuning {

	public static class DeepParameterTuningProblem extends AbstractProblem {
		public static int numEvaluations=0;	
		public static final int[] genotype = {0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,2,2,2,1,0,2,2,0,0,0,0,0,2,2,1,1,1,1,0,0,1,1,1,2,31,0,0,1,1,0,0,0,0,0};	
		private static final int numObjectives=2;
		public DeepParameterTuningProblem() {
			super(genotype.length,numObjectives);
		}

		public Solution newSolution() {
			Solution solution = new Solution(getNumberOfVariables(),getNumberOfObjectives());
			for(int i=0; i < getNumberOfVariables(); i++) {
				solution.setVariable(i, new RealVariable(genotype[i], 0.0, 64.0));
			}
			return solution;
		}

		public void evaluate(Solution solution) {
			double[] objectiveResults= new double[numObjectives];
			
			//A Hack to ensure the first generation contains an instance of the original software		
			if(numEvaluations==0){
				for(int i=0; i< solution.getNumberOfVariables(); i++){
					((RealVariable) solution.getVariable(i)).setValue(genotype[i]);
				}
			}
			numEvaluations++;

			double[] values = getValues(solution);

			String[] cmd = new String[values.length + 2];
			cmd[0]="/bin/bash";
			cmd[1]="load_and_evaluation.bsh";
			for(int i=0; i<values.length; i++){
				cmd[i+2]=Integer.toString((int) values[i]);			
			}
           	
			double time_value=60;
                        double accuracy_value=0;

			try{
				Process pr = Runtime.getRuntime().exec(cmd);
				BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));
				String inputLine;
    				while ((inputLine = in.readLine()) != null) {
				
					String[] parts = inputLine.split(",");
					time_value=Double.parseDouble(parts[0]);
					accuracy_value=Double.parseDouble(parts[1]);
				}
			} catch (Exception e){}
			
			
			System.out.println(String.valueOf(numEvaluations -1)+" "+Arrays.toString(cmd)+" "+String.valueOf(time_value)+" "+String.valueOf(accuracy_value));
			objectiveResults[0]=(time_value);
			objectiveResults[1]=10000/(accuracy_value + 0.0001);

			solution.setObjectives(objectiveResults);
		}

		public static double[] getValues(Solution solution) {
			double [] toReturn = new double [solution.getNumberOfVariables()];
			for(int i=0; i < solution.getNumberOfVariables(); i++){
				toReturn[i] = ((RealVariable)solution.getVariable(i)).getValue();
			}
			return toReturn;
		}
	}

	public static void main(String[] args) {
		NondominatedPopulation result = new Executor()
                        .withProblemClass(DeepParameterTuningProblem.class)
                        .withAlgorithm("NSGAII")
             		.withMaxEvaluations(3000)
                        .withCheckpointFrequency(5)
                        .withCheckpointFile(new File("example.state"))
                        .run();
		
		System.out.println();
		System.out.println();		
		//Display the results
		for(Solution solution1 : result) {
			double values[] = DeepParameterTuningProblem.getValues(solution1);
			int valuesInt[] = new int[values.length];
			for(int i=0; i<values.length; i++){
				valuesInt[i] = ((int) values[i]);
			}

			double time_value=solution1.getObjective(0);
			double percentage_accuracy=(10000 / solution1.getObjective(1)) - 0.0001;			

			System.out.format("%.6f	%.6f %s%n",
				time_value,
				percentage_accuracy,
				Arrays.toString(valuesInt));
		}
		
	}
}
