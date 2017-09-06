package ude.master.thesis.stance_detection.ml;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVReader;

import ude.master.thesis.stance_detection.util.PPDBProcessor;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FerreiraFeaturesClassifier {

	private static FileHashMap<String, ArrayList<Double>> trainRootDist;
	private static FileHashMap<String, ArrayList<Double>> testRootDist;
	private static FileHashMap<String, Double> trainPPDB;
	private static FileHashMap<String, Double> testPPDB;
	private static FileHashMap<String, Integer> trainNeg;
	private static FileHashMap<String, Integer> testNeg;
	private static Map<Integer, String> trainIdBodyMap;
	private static List<List<String>> trainingStances;
	private static HashMap<Integer, String> testIdBodyMap;
	private static List<List<String>> testStances;
	private static FileHashMap<String, Double> trainW2VSim;
	private static FileHashMap<String, Double> testW2VSim;
	private static FileHashMap<String, ArrayList<Integer>> trainSVO;
	private static FileHashMap<String, ArrayList<Integer>> testSVO;

	public static void main(String[] args) throws IOException {

		// Saving maps
		/*
		 * try { saveRootDistFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/train_features/train_rootdist.csv",
		 * "C:/thesis_stuff/features/train_features/train_rootdist");
		 * saveRootDistFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/test_features/test_rootdist.csv",
		 * "C:/thesis_stuff/features/test_features/test_rootdist");
		 * 
		 * savePPDBFeaturesAsHashFiles(
		 * "C:/thesis_stuff/features/train_features/train_hung_ppdb.csv",
		 * "C:/thesis_stuff/features/train_features/train_hung_ppdb");
		 * savePPDBFeaturesAsHashFiles(
		 * "C:/thesis_stuff/features/test_features/test_hung_ppdb.csv",
		 * "C:/thesis_stuff/features/test_features/test_hung_ppdb");
		 * 
		 * saveNegFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/train_features/train_neg_features.csv",
		 * "C:/thesis_stuff/features/train_features/train_neg_features");
		 * saveNegFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/test_features/test_neg_features.csv",
		 * "C:/thesis_stuff/features/test_features/test_neg_features");
		 * 
		 * saveWord2VecFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/train_features/train_w2v_mean_sim_formula_features.csv",
		 * "C:/thesis_stuff/features/train_features/train_w2v_mean_sim_formula_features"
		 * ); saveWord2VecFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/test_features/test_w2v_mean_sim_formula_features.csv",
		 * "C:/thesis_stuff/features/test_features/test_w2v_mean_sim_formula_features"
		 * ); } catch (ObjectExistsException | ClassNotFoundException |
		 * VersionMismatchException | IOException e) { // TODO Auto-generated
		 * catch block e.printStackTrace(); }
		 */
		try {
			saveSVOFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_svo_nosvo_features.csv", "C:/thesis_stuff/features/train_features/train_svo_nosvo_features");
			saveSVOFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_svo_nosvo_features.csv", "C:/thesis_stuff/features/test_features/test_svo_nosvo_features");
		} catch (ObjectExistsException | ClassNotFoundException | VersionMismatchException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		
		try {
			trainRootDist = loadRootDistFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_rootdist");
			testRootDist = loadRootDistFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_rootdist");

			trainPPDB = loadPPDBFeaturesAsHashFiles("C:/thesis_stuff/features/train_features/train_hung_ppdb");
			testPPDB = loadPPDBFeaturesAsHashFiles("C:/thesis_stuff/features/test_features/test_hung_ppdb");

			trainNeg = loadNegFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_neg_features");
			testNeg = loadNegFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_neg_features");

			trainW2VSim = loadWord2VecFeaturesAsHashFile(
					"C:/thesis_stuff/features/train_features/train_w2v_mean_sim_formula_features");
			testW2VSim = loadWord2VecFeaturesAsHashFile(
					"C:/thesis_stuff/features/test_features/test_w2v_mean_sim_formula_features");
			
			trainSVO = loadSVOFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_svo_nosvo_features");
			testSVO = loadSVOFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_svo_nosvo_features");
		} catch (ObjectExistsException | ClassNotFoundException | VersionMismatchException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Create training ARFF file
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, "resources/data/train_stances.csv",
				"resources/data/summ_train_bodies.csv", "resources/data/test_data/competition_test_stances.csv",
				"resources/data/test_data/summ_competition_test_bodies.csv");

		trainIdBodyMap = sddr.getTrainIdBodyMap();
		trainingStances = sddr.getTrainStances();

		testIdBodyMap = sddr.getTestIdBodyMap();
		testStances = sddr.getTestStances();

		saveARFF(trainingStances, trainRootDist, trainPPDB, trainNeg, trainW2VSim,trainSVO,
				"C:/thesis_stuff/features/train_ferreira_with_svo_minus_one.arff");
		saveARFF(testStances, testRootDist, testPPDB, testNeg, testW2VSim, testSVO,
				"C:/thesis_stuff/features/test_ferreira_with_svo_minus_one.arff");

	}

	/**
	 * 
	 * @param stancesData
	 * @param rootDist
	 * @param ppdbhung
	 * @param negData
	 * @param simData
	 * @param filepath
	 * @throws IOException
	 */
	private static void saveARFF(List<List<String>> stancesData, FileHashMap<String, ArrayList<Double>> rootDist,
			FileHashMap<String, Double> ppdbhung, FileHashMap<String, Integer> negData,
			FileHashMap<String, Double> simData, FileHashMap<String, ArrayList<Integer>> svoData, String filepath) throws IOException {
		ArrayList<Attribute> attributes = new ArrayList<>();

		String stances[] = new String[] { "agree", "disagree", "discuss" };
		List<String> stanceValues = Arrays.asList(stances);

		attributes.add(new Attribute("root_dis_ref"));
		attributes.add(new Attribute("root_dis_disc"));
		attributes.add(new Attribute("ppdb"));
		attributes.add(new Attribute("neg"));
		attributes.add(new Attribute("w2v_sim_mean"));
		
		//add svo attributes
		for(int i = 0; i < 12; i++){
			attributes.add(new Attribute("svo_" + i));
		}
		

		attributes.add(new Attribute("class", stanceValues));

		Instances instances = new Instances("fnc-1-Ferreira", attributes, 1000);
		instances.setClassIndex(attributes.size() - 1);

		for (List<String> s : stancesData) {
			String stance = s.get(2);
			if (!stance.equals("unrelated")) {

				Double rootdistRef = rootDist.get(s.get(1)).get(2);
				Double rootdistDis = rootDist.get(s.get(1)).get(3);

				Double ppdb = ppdbhung.get(s.get(0) + s.get(1));

				Integer neg = negData.get(s.get(0) + s.get(1));

				Double sim = simData.get(s.get(0) + s.get(1));
				
				ArrayList<Integer> svos = svoData.get(s.get(0) + s.get(1));
				
				

				DenseInstance instance = new DenseInstance(attributes.size());
				instance.setDataset(instances);
				instance.setValue(instances.attribute("root_dis_ref"), rootdistRef);
				instance.setValue(instances.attribute("root_dis_disc"), rootdistDis);
				instance.setValue(instances.attribute("ppdb"), ppdb);
				instance.setValue(instances.attribute("neg"), neg);
				instance.setValue(instances.attribute("w2v_sim_mean"), sim);
				
				//add svo features vector
				for(int i = 0; i < 12; i++){
					instance.setValue(instances.attribute("svo_" + i), svos.get(i));
				}

				instance.setClassValue(stance);

				instances.add(instance);
			}
		}

		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new java.io.File(filepath));
		saver.writeBatch();
	}

	/**
	 * This represent 2 features: the distance from the root of a sentence to a
	 * refuting/discussing word saving the map as [
	 * <body_id>,["ref_RootDist","disc_RootDist","ref_avg","disc_avg"]]
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static void saveRootDistFeaturesAsHashFile(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> rootDistData = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {

			ArrayList<Double> values = new ArrayList<>();
			values.add(Double.valueOf(line[1]));
			values.add(Double.valueOf(line[2]));
			values.add(Double.valueOf(line[3]));
			values.add(Double.valueOf(line[4]));

			rootDistData.put(line[0], values);
		}
		reader.close();

		// saving the map file
		rootDistData.save();
		rootDistData.close();

	}

	/**
	 * 
	 * @param hashFileName
	 * @return
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static FileHashMap<String, ArrayList<Double>> loadRootDistFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Double>> rootDistData = new FileHashMap<String, ArrayList<Double>>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return rootDistData;

	}

	/**
	 * saves the ppdb hugarian score as a map <<title + body texts>,<score
	 * value>>
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static void savePPDBFeaturesAsHashFiles(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> ppdbData = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			ppdbData.put(line[0] + line[1], Double.valueOf(line[3]));
		}
		reader.close();

		// saving the map file
		ppdbData.save();
		ppdbData.close();

	}

	/**
	 * 
	 * @param hashFileName
	 * @return
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static FileHashMap<String, Double> loadPPDBFeaturesAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> ppdbData = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return ppdbData;
	}

	/**
	 * saves the ppdb hugarian score as a map <<title + body texts>,<neg value>>
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static void saveNegFeaturesAsHashFile(String csvfilePath, String hashFileName) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, Integer> negData = new FileHashMap<String, Integer>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			negData.put(line[0] + line[1], Integer.valueOf(line[3]));
		}
		reader.close();

		// saving the map file
		negData.save();
		negData.close();
	}

	/**
	 * 
	 * @param hashFileName
	 * @return
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static FileHashMap<String, Integer> loadNegFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Integer> negData = new FileHashMap<String, Integer>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return negData;
	}

	/**
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws IOException 
	 * @throws VersionMismatchException 
	 * @throws ClassNotFoundException 
	 * @throws ObjectExistsException 
	 * @throws FileNotFoundException 
	 */
	public static void saveSVOFeaturesAsHashFile(String csvfilePath, String hashFileName) throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, ArrayList<Integer>> svoData = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);

		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			svoData.put(line[0] + line[1], PPDBProcessor.getIntList(line[3]));
		}
		reader.close();

		// saving the map file
		svoData.save();
		svoData.close();
	}
	
	/**
	 * 
	 * @param hashFileName
	 * @return
	 * @throws FileNotFoundException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 * @throws IOException
	 */
	public static FileHashMap<String, ArrayList<Integer>> loadSVOFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Integer>> svoData = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return svoData;
	}

	/**
	 * saves the ppdb hugarian score as a map <<title + body texts>,<sim value>>
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static void saveWord2VecFeaturesAsHashFile(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> w2vSimData = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);

		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			w2vSimData.put(line[0] + line[1], Double.valueOf(line[3]));
		}
		reader.close();

		// saving the map file
		w2vSimData.save();
		w2vSimData.close();
	}

	/**
	 * 
	 * @param hashFileName
	 * @return
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static FileHashMap<String, Double> loadWord2VecFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Double> w2vSimData = new FileHashMap<String, Double>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return w2vSimData;
	}
}
