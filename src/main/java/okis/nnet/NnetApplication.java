package okis.nnet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class NnetApplication {
	
	private static Double[] X1 = {
			1.169650579,
			1.092518429,
			0.968090859,
			1.11218082,
			0.901668248,
			0.992068878

	},		
			X2 = {
			1.43863443,
			1.383979027,
			1.49974416,
			1.202438427,
			1.491568729,
			1.24215481
	}, 
			Y = {
			0.542999236,
			1.74085382,
			1.941561179,
			0.441349288,
			0.409252791,
			0.385815401,
	};
	
	public static void main(String[] args) {
		NeuralNetwork ann = new MultiLayerPerceptron(2, 8, 8, 8, 8, 1); 
		MomentumBackpropagation learningRule = (MomentumBackpropagation) ann.getLearningRule();
		learningRule.setLearningRate(0.2);
        learningRule.setMaxError(0.01);
        learningRule.setMaxIterations(5000);
		
		int inputSize = 2;
		int outputSize = 1;
		DataSet ds = new DataSet(inputSize, outputSize);
		
		File file = new File("D:\\JavaProjects\\NeuralNetwork2221\\src\\okis\\neuralnetwork\\input.txt");
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			String st;
			while ((st = br.readLine()) != null) {
				String[] result = st.split("\t");			    
				ds.addRow(new DataSetRow(
						new double[] {Double.valueOf(result[0]), Double.valueOf(result[1])}, 
						new double[] {Double.valueOf(result[2])}
						));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("Size of testing rows: "+ds.getRows().size());
		
//		BackPropagation backPropagation = new BackPropagation();
//		backPropagation.setMaxIterations(10000);
//		backPropagation.setMaxError(0.01);
		ann.learn(ds);
		
		for(int i=0;i<X1.length;i++) {
			ann.setInput(X1[i], X2[i]);
			ann.calculate();
			double[] networkOutputOne = ann.getOutput();
			System.out.println("Testing: {"+X1[i]+", "+X2[i]+"}.\tExcepted: "+Y[i]+".\tResult: "+networkOutputOne[0]);
		}
	}

}
