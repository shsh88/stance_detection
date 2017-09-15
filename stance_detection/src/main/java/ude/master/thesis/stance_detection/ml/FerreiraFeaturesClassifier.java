package ude.master.thesis.stance_detection.ml;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;
import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVReader;

import ude.master.thesis.stance_detection.main.ClassifierTools;
import ude.master.thesis.stance_detection.processor.FeatureExtractor;
import ude.master.thesis.stance_detection.processor.Lemmatizer;
import ude.master.thesis.stance_detection.util.PPDBProcessor;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import ude.master.thesis.stance_detection.wordembeddings.DocToVec;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;

import weka.core.converters.ConverterUtils.DataSource;

public class FerreiraFeaturesClassifier {
	final static Logger logger = Logger.getLogger(FerreiraFeaturesClassifier.class);

	private static FileHashMap<String, ArrayList<Double>> trainRootDist;
	private static FileHashMap<String, ArrayList<Double>> testRootDist;
	private static FileHashMap<String, double[]> trainPPDB;
	private static FileHashMap<String, double[]> testPPDB;
	private static FileHashMap<String, Integer> trainNeg;
	private static FileHashMap<String, Integer> testNeg;
	private static List<List<String>> trainingStances;
	private static List<List<String>> testStances;
	private static FileHashMap<String, Double> trainW2VSim;
	private static FileHashMap<String, Double> testW2VSim;
	private static FileHashMap<String, ArrayList<Integer>> trainSVO;
	private static FileHashMap<String, ArrayList<Integer>> testSVO;
	private static HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap;
	private static HashMap<Integer, Map<Integer, String>> testSummIdBoyMap;
	private static Lemmatizer lemmatizer;
	private static DocToVec doc2vec;

	private static Instances trainingInstances;

	private static Instances testInstances;

	public static void main(String[] args) throws Exception {

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
		 * "C:/thesis_stuff/features/train_features/train_hung_ppdb_with_stopwords.csv",
		 * "C:/thesis_stuff/features/train_features/train_hung_ppdb_with_stopwords"
		 * ); savePPDBFeaturesAsHashFiles(
		 * "C:/thesis_stuff/features/test_features/test_hung_ppdb_with_stopwords.csv",
		 * "C:/thesis_stuff/features/test_features/test_hung_ppdb_with_stopwords"
		 * );
		 * 
		 * saveNegFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/train_features/train_neg_features.csv",
		 * "C:/thesis_stuff/features/train_features/train_neg_features");
		 * saveNegFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/test_features/test_neg_features.csv",
		 * "C:/thesis_stuff/features/test_features/test_neg_features");
		 * 
		 * saveWord2VecFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/train_features/train_w2v_sim_features.csv",
		 * "C:/thesis_stuff/features/train_features/train_w2v_sim_features");
		 * saveWord2VecFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/test_features/test_w2v_sim_features.csv",
		 * "C:/thesis_stuff/features/test_features/test_w2v_sim_features"); }
		 * catch (ObjectExistsException | ClassNotFoundException |
		 * VersionMismatchException | IOException e) { // TODO Auto-generated
		 * catch block e.printStackTrace(); }
		 * 
		 * try { saveSVOFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/train_features/train_svo_nosvo_features.csv",
		 * "C:/thesis_stuff/features/train_features/train_svo_nosvo_features");
		 * saveSVOFeaturesAsHashFile(
		 * "C:/thesis_stuff/features/test_features/test_svo_nosvo_features.csv",
		 * "C:/thesis_stuff/features/test_features/test_svo_nosvo_features"); }
		 * catch (ObjectExistsException | ClassNotFoundException |
		 * VersionMismatchException e1) { // TODO Auto-generated catch block
		 * e1.printStackTrace(); }
		 */

		try {
			trainRootDist = loadRootDistFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_rootdist");
			testRootDist = loadRootDistFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_rootdist");

			trainPPDB = loadPPDBFeaturesAsHashFiles(
					"C:/thesis_stuff/features/train_features/train_hung_ppdb_with_stopwords");
			testPPDB = loadPPDBFeaturesAsHashFiles(
					"C:/thesis_stuff/features/test_features/test_hung_ppdb_with_stopwords");

			trainNeg = loadNegFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_neg_features");
			testNeg = loadNegFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_neg_features");

			trainW2VSim = loadWord2VecFeaturesAsHashFile(
					"C:/thesis_stuff/features/train_features/train_w2v_sim_features");
			testW2VSim = loadWord2VecFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_w2v_sim_features");

			trainSVO = loadSVOFeaturesAsHashFile("C:/thesis_stuff/features/train_features/train_svo_nosvo_features");
			testSVO = loadSVOFeaturesAsHashFile("C:/thesis_stuff/features/test_features/test_svo_nosvo_features");
		} catch (ObjectExistsException | ClassNotFoundException | VersionMismatchException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Create training ARFF file
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				"resources/data/train_stances_preprocessed.csv", "resources/data/train_bodies_preprocessed_summ.csv",
				"resources/data/test_data/test_stances_preprocessed.csv",
				"resources/data/test_data/test_bodies_preprocessed_summ.csv");

		trainingSummIdBoyMap = sddr.readSummIdBodiesMap(new File("resources/data/train_bodies_preprocessed_summ.csv"));
		testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File("resources/data/test_data/test_bodies_preprocessed_summ.csv"));

		trainingStances = sddr.getTrainStances();
		System.out.println(trainingStances.size());
		

		testStances = sddr.getTestStances();
		System.out.println(testStances.size());
		
		lemmatizer = new Lemmatizer();

		String trainArffFile = "C:/thesis_stuff/features/train_ferreira_with_svo_minus_one_doc2vec100_BoW1000_150917_0854.arff";
		String testArffFile = "C:/thesis_stuff/features/test_ferreira_with_svo_minus_one_doc2vec100_BoW1000_150917_0854.arff";
		
		saveARFF(trainingStances, trainingSummIdBoyMap, trainRootDist, trainPPDB, trainNeg, trainW2VSim, trainSVO,
				trainArffFile);
		saveARFF(testStances, testSummIdBoyMap, testRootDist, testPPDB, testNeg, testW2VSim, testSVO, testArffFile);
		
		filterAndClassify(trainArffFile, testArffFile);
	}

	private static void filterAndClassify(String trainArffFile, String testArffFile) throws Exception {
		trainingInstances = readInstancesFromArff(trainArffFile);
		testInstances = readInstancesFromArff(testArffFile);
		LibLINEAR classifier = new LibLINEAR();
	
		classifier.setOptions(weka.core.Utils.splitOptions("-S 6 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000"));
		ClassifierTools ct = new ClassifierTools(trainingInstances, testInstances, classifier);
		ct.applyBoWFilter();
		ct.applyAttributSelectionFilter();
		ct.saveInstancesToArff("ferr_doc2vec100_BoW1000_1200f_attSV");
		
		
		//ct.evaluateWithCrossValidation("C:/thesis_stuff/results/"+ "modi_ferr");
		ct.train(true, "C:/thesis_stuff/results/"+ "modi_ferr_doc2vec100_BoW1000_1200f_attSV");
		ct.evaluateOnTestset("C:/thesis_stuff/results/"+ "modi_ferr_doc2vec100_BoW1000_1200f_attSV");
	}

	private static Instances readInstancesFromArff(String trainArffFile) throws Exception {
		DataSource dataSource = new DataSource(trainArffFile);
		Instances instances = dataSource.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}

	/**
	 * 
	 * @param stancesData
	 * @param trainingSummIdBoyMap2
	 * @param rootDist
	 * @param trainPPDB2
	 * @param negData
	 * @param simData
	 * @param filepath
	 * @throws IOException
	 */
	private static void saveARFF(List<List<String>> stancesData, HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			FileHashMap<String, ArrayList<Double>> rootDist, FileHashMap<String, double[]> trainPPDB2,
			FileHashMap<String, Integer> negData, FileHashMap<String, Double> simData,
			FileHashMap<String, ArrayList<Integer>> svoData, String filepath) throws IOException {
		ArrayList<Attribute> attributes = new ArrayList<>();

		String stances[] = new String[] { "agree", "disagree", "discuss" };
		List<String> stanceValues = Arrays.asList(stances);

		attributes.add(new Attribute("boby_Summ", (List<String>) null));

		for (int i = 0; i < 10; i++)
			attributes.add(new Attribute("root_dis_ref_" + i));

		for (int i = 0; i < 10; i++)
			attributes.add(new Attribute("root_dis_disc_" + i));

		for (int i = 0; i < 10; i++)
			attributes.add(new Attribute("ppdb_" + i));

		attributes.add(new Attribute("neg"));
		attributes.add(new Attribute("w2v_sim_sum"));

		// add svo attributes
		for (int i = 0; i < 12; i++) {
			attributes.add(new Attribute("svo_" + i));
		}

		for (int i = 0; i < 100; i++){
			attributes.add(new Attribute("t_d2vec_" + i));
		}
		for (int i = 0; i < 100; i++){
			attributes.add(new Attribute("b_d2vec_" + i));
		}
		attributes.add(new Attribute("stance_class", stanceValues));

		Instances instances = new Instances("fnc-1-Ferreira", attributes, 1000);
		instances.setClassIndex(attributes.size() - 1);

		for (List<String> s : stancesData) {
			String stance = s.get(2);
			if (!stance.equals("unrelated")) {

				ArrayList<Double> rootdist = rootDist.get(s.get(1));

				double[] ppdb = trainPPDB2.get(s.get(0) + s.get(1));

				Integer neg = negData.get(s.get(0) + s.get(1));

				Double sim = simData.get(s.get(0) + s.get(1));

				ArrayList<Integer> svos = svoData.get(s.get(0) + s.get(1));

				DenseInstance instance = new DenseInstance(attributes.size());
				instance.setDataset(instances);
				
				String part1 = FeatureExtractor.getLemmatizedCleanStr(summIdBoyMap.get(Integer.valueOf(s.get(1))).get(1));
				String part2 = FeatureExtractor.getLemmatizedCleanStr(summIdBoyMap.get(Integer.valueOf(s.get(1))).get(3));
				
				instance.setValue(instances.attribute("boby_Summ"), part1
						+ " " + part2);
				
				for (int i = 0; i < 10; i++)
					instance.setValue(instances.attribute("root_dis_ref_" + i), rootdist.get(i));
				for (int i = 0; i < 10; i++)
					instance.setValue(instances.attribute("root_dis_disc_" + i), rootdist.get(i + 10));

				for (int i = 0; i < 10; i++) {
					instance.setValue(instances.attribute("ppdb_" + i), ppdb[i]);
				}

				instance.setValue(instances.attribute("neg"), neg);
				instance.setValue(instances.attribute("w2v_sim_sum"), sim);

				// add svo features vector
				for (int i = 0; i < 12; i++) {
					instance.setValue(instances.attribute("svo_" + i), svos.get(i));
				}
				
				if(doc2vec == null)
					doc2vec = new DocToVec();
				
				double[] tVec = doc2vec.getTitleParagraphVecByLabel(s.get(0));
				for (int i = 0; i < 100; i++){
					instance.setValue(instances.attribute("t_d2vec_" + i), tVec[i]);
				}
				
				double[] bVec = doc2vec.getBodyParagraphVecByLabel(s.get(1));
				for (int i = 0; i < 100; i++){
					instance.setValue(instances.attribute("b_d2vec_" + i), bVec[i]);
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
	 * <body_id>,["ref_RootDist","disc_RootDist"]] These 2 features are
	 * calculated for each sentence in the body (6 from the beginning and 4 at
	 * last) So the feature vector is of length 10
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
			for (int i = 1; i <= 20; i++)
				values.add(Double.valueOf(line[i]));

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
		FileHashMap<String, double[]> ppdbData = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		int lineNo = 0;

		while ((line = reader.readNext()) != null) {
			double[] featureVecor = new double[10];
			for (int i = 3; i <= 12; i++) {
				Double v = Double.valueOf(line[i].split("\\|")[0]);
				featureVecor[i - 3] = v;
			}
			ppdbData.put(line[0] + line[1], featureVecor);
			lineNo++;
			if (lineNo % 1000 == 0)
				System.out.println(lineNo);
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
	public static FileHashMap<String, double[]> loadPPDBFeaturesAsHashFiles(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, double[]> ppdbData = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return ppdbData;
	}

	/**
	 * saves the ppdb hugarian score as a map <<title + body texts>,<neg value>>
	 * count feature
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
	public static void saveSVOFeaturesAsHashFile(String csvfilePath, String hashFileName) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
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
