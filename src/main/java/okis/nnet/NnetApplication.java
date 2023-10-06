package okis.nnet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationClippedLinear;
import org.encog.engine.network.activation.ActivationCompetitive;
import org.encog.engine.network.activation.ActivationElliott;
import org.encog.engine.network.activation.ActivationElliottSymmetric;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationGaussian;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationRamp;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationSteepenedSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 * Class for test optimal neural network architecture
 */
public class NnetApplication {
	/**
	 * Row counter of filling Excel table
	 */
	static int rowNum = 0;
	
	/**
	 * Column counter of filling Excel table
	 */
	static int columnNum = 0;
	
	public static void main(String[] args) {
		/**
		 * List of count layers in testing neural network
		 */
		int[] layers 	= new int[] {1, 2, 4};
		
		/**
		 * List of count neurons by hidden layer in testing neural network
		 */
		int[] neurons 	= new int[] {2, 4, 8};
		
		/**
		 * Input/output values
		 */
		List<double[]> input = new ArrayList<double[]>();
		List<Double>   idealOutput = new ArrayList<Double>();
		
		/**
		 * Testing activation functions
		 */
		ActivationFunction[] activationFunctions = new ActivationFunction[] {
			new ActivationTANH(),
			new ActivationSigmoid(),
			new ActivationSIN(),
			new ActivationReLU(),
			new ActivationLinear()
//			new ActivationRamp(),
//			new ActivationLOG(),
//			new ActivationGaussian(),
//			new ActivationElliottSymmetric(),
//			new ActivationElliott(),
//			new ActivationCompetitive(),
//			new ActivationClippedLinear()
		};
		
		/**
		 * File, where stored input (x1, x2) and ideal output (y) values
		 */
		File file = new File("F:\\NN\\input.txt");
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(file));
			String st;
			while ((st = br.readLine()) != null) {
				String[] result = st.split("\t");	
				input.add(new double[] {Double.valueOf(result[0]), Double.valueOf(result[1])});
				idealOutput.add(Double.valueOf(result[2]));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		double[][] in_arr 	= new double[input.size()][3];
		double[][] out_arr 	= new double[idealOutput.size()][2];

		//convert to double[][] array
		for (int i = 0; i < input.size(); i++) {
        	in_arr[i][0] 	= input.get(i)[0];
        	in_arr[i][1] 	= input.get(i)[1];
        	out_arr[i][0] 	= idealOutput.get(i).doubleValue();
        }
		
		// create training data
		MLDataSet trainingSet = new BasicMLDataSet(in_arr, out_arr);
		
		try {
			//Writing result to Excel table
			XSSFWorkbook workbook = new XSSFWorkbook(); 
		
			XSSFSheet sheet = workbook.createSheet("Result");
			
			for(int layer: layers) {
				for(int neuron: neurons) {	
					for(ActivationFunction activationFunction : activationFunctions) {
						for(ActivationFunction subActivationFunction : activationFunctions) {
						// create a neural network, without using a factory
						BasicNetwork network = new BasicNetwork();
						network.addLayer(new BasicLayer(null,true,2));
						
						//set activation function for hidden layer
						for(int i=0;i<layer;i++) {
							Class<?> clazz = Class.forName(subActivationFunction.getClass().toString().split(" ")[1]);
							Constructor<?> ctor = clazz.getConstructors()[0];
							network.addLayer(new BasicLayer((ActivationFunction) ctor.newInstance(),true, neuron));
						}
						
						//set activation function for output layer
						Class<?> clazz = Class.forName(activationFunction.getClass().toString().split(" ")[1]);
						Constructor<?> ctor = clazz.getConstructors()[0];
						network.addLayer(new BasicLayer((ActivationFunction) ctor.newInstance(),false,1));
						network.getStructure().finalizeStructure();
						network.reset();
						
						// train the neural network
						final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
				
						int epoch = 1;
						long startTime = System.nanoTime();
			
						do {
							train.iteration();
							epoch++;
						} while(epoch<1000);
						
						train.finishTraining();
						long endTime = System.nanoTime();
						
						
						System.out.println(layer+" layers by "+neuron+" neurons. Error: "+train.getError()+"%. \tTraining time: "+(endTime-startTime)+" ns");
						
						PrintWriter actual;
						try {
							
							//Save result by everyone neural network to individual txt file and column of Excel table
							String subActivationName 	= subActivationFunction.getClass().toString().split("Activation")[1],
								   activationName 		= activationFunction.getClass().toString().split("Activation")[1];
							
							
							actual 	= new PrintWriter("F:\\NN\\"+layer+" "+subActivationName+"actual"+neuron+" "+activationName+".txt", "UTF-8");
							
							Row row = nextRow(columnNum, sheet);
							Cell cell = row.createCell(columnNum);
							cell.setCellValue((String)(layer+" "+subActivationName+"y"+neuron+" "+activationName));
							
							for(MLDataPair pair: trainingSet ) {
								final MLData output = network.compute(pair.getInput());
								actual.println(String.valueOf(output.getData(0)).replace('.', ','));
								
								row = nextRow(columnNum, sheet);
								
								cell = row.createCell(columnNum);
								cell.setCellValue((Double)output.getData(0));
							}
							columnNum++;
							rowNum = 0;
							actual.close();
						} catch (FileNotFoundException e) {
							e.printStackTrace();
						} catch (UnsupportedEncodingException e) {
							e.printStackTrace();
						}
						Encog.getInstance().shutdown();
						}
					}
				}
			}
			
			//write data to Excel table
			try {
				FileOutputStream out = new FileOutputStream(new File("F:\\NN\\result.xlsx"));
				workbook.write(out);
				out.close();
				System.out.println("result.xlsx written successfully on disk.");
			} 
			catch (Exception e) {
				e.printStackTrace();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static Row nextRow(int columnNum, XSSFSheet sheet) {
		if(columnNum == 0) {
			return sheet.createRow(rowNum++);
		}
		return sheet.getRow(rowNum++);
	}
}
