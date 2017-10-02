package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class Word2VecDataGenerator2 {

	private static Word2Vec vec;
	private static Lemmatizer lemm;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		//vec = loadGoogleNewsVec();
		//lemm = new Lemmatizer();
		String txt = "Kim Jong-un has broken both of his ankles and is now in the hospital after undergoing "
				+ "surgery, a report in a South Korean newspaper claims. The North Korean leader has "
				+ "been missing for more than three weeks, fueling speculation about what could cause his "
				+ "unusual disappearance from the public eye. This rumor seems to confirm what North Korean "
				+ "state media had said on Thursday, when state broadcaster Korean Central Television "
				+ "reported that Kim was \"not feeling well,\" and was suffering from an \"uncomfortable "
				+ "physical condition.\" Have something to add to this story? Share it in the comments.";

		// String headline = "ISIL Beheads American Photojournalist in Iraq";
		String headline = "American Photojournalist";
		String body = "James Foley, an American journalist who went missing in Syria more than a year ago, "
				+ "has reportedly been executed by the Islamic State, a militant group formerly known as ISIS. "
				+ "Video and photos purportedly of Foley emerged on Tuesday. A YouTube video -- entitled \"\"A "
				+ "Message to #America (from the #IslamicState)\"\" -- identified a man on his knees as"
				+ " \"\"James Wright Foley,\"\" and showed his execution. This is a developing story. "
				+ "Check back here for updates.";

		// System.out.println("\"" + FeatureExtractor.clean(txt) + "\"");

		// System.out.println("\"" + FeatureExtractor.clean("?").trim() + "\"");

		/*
		 * BufferedWriter out = new BufferedWriter(new
		 * FileWriter("resources/words_not_processed.txt"));
		 * out.write("=========================="); out.newLine();
		 * 
		 * Lemmatizer lemm = new Lemmatizer(); List<String> lemmas =
		 * lemm.lemmatize(body); System.out.println(); INDArray zeros =
		 * Nd4j.zeros(1, 300); INDArray identity = Nd4j.ones(1, 300);
		 * 
		 * System.out.println(identity.shapeInfoToString());
		 * 
		 * ArrayList<INDArray> m = new ArrayList<>(); for (String tok : lemmas)
		 * { INDArray tokVec; //
		 * if(!FeatureExtractor.clean(tok).trim().isEmpty()){ if
		 * (vec.hasWord(FeatureExtractor.clean(tok).trim())) { tokVec =
		 * Nd4j.create(vec.getWordVector(tok.trim()));
		 * System.out.println(tokVec.shapeInfoToString());
		 * System.out.println(tok); } else tokVec = identity; out.write(tok +
		 * ": \n" + tokVec.toString()); m.add(tokVec); // System.out.println(tok
		 * + ": "+ tokVec.toString());
		 * 
		 * } out.flush(); out.close();
		 * 
		 * INDArray result = m.stream().reduce(identity, INDArray::mul);
		 * 
		 * System.out.println(result.toString());
		 */

		/*
		 * ArrayList<INDArray> m = new ArrayList<>(); INDArray originalArray =
		 * Nd4j.linspace(1,150,300).reshape('c',1,300);
		 * System.out.println(originalArray); m.add(originalArray);
		 * 
		 * originalArray = Nd4j.linspace(-150,-1,300).reshape('c',1,300);
		 * System.out.println(originalArray); m.add(originalArray);
		 * 
		 * originalArray = Nd4j.linspace(-333,5,300).reshape('c',1,300);
		 * System.out.println(originalArray); m.add(originalArray);
		 * 
		 * originalArray = Nd4j.linspace(0,10,300).reshape('c',1,300);
		 * System.out.println(originalArray); m.add(originalArray);
		 * 
		 * originalArray = Nd4j.linspace(4,20,4).reshape('c',1,4);
		 * System.out.println(originalArray); //m.add(originalArray);
		 * 
		 * INDArray identity = Nd4j.ones(1, 300); INDArray result =
		 * m.stream().reduce(identity, INDArray::mul);
		 * System.out.println(result.shapeInfoToString());
		 * System.out.println(result.toString());
		 */

		// generateWord2VecData();
		saveWord2VecFeaturesAsHashFile(ProjectPaths.CSV_TRAIN_W2V_SUM_SIM_PARTS2, ProjectPaths.W2V_SIM_ADD_TRAIN2);
		saveWord2VecFeaturesAsHashFile(ProjectPaths.CSV_TEST_W2V_SUM_SIM_PARTS2, ProjectPaths.W2V_SIM_ADD_TEST2);
	}

	private static void generateWord2VecData()
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2);

		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2));
		List<List<String>> trainingStances = sddr.getTrainStances();

		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2));
		List<List<String>> testStances = sddr.getTestStances();

		generateAddWord2VecAndSave(trainingSummIdBoyMap, trainingStances, testSummIdBoyMap, testStances,
				ProjectPaths.BODIES_W2V_SUM2, ProjectPaths.TITLES_W2V_SUM2);

		generateWord2VecAddSimFeaturesAndSave(trainingStances, ProjectPaths.CSV_TRAIN_W2V_SUM_SIM_PARTS2);
		generateWord2VecAddSimFeaturesAndSave(testStances, ProjectPaths.CSV_TEST_W2V_SUM_SIM_PARTS2);

	}

	private static void generateAddWord2VecAndSave(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			List<List<String>> trainingStances, HashMap<Integer, Map<Integer, String>> testSummIdBoyMap,
			List<List<String>> testStances, String bodiesVecPath, String titlesVecPath) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		// getting titles vectors
		FileHashMap<String, double[]> trainTVecsMap = new FileHashMap<String, double[]>(titlesVecPath,
				FileHashMap.FORCE_OVERWRITE);
		Set<String> titles = new HashSet<>();

		for (List<String> s : trainingStances) {
			titles.add(s.get(0));

		}
		for (List<String> s : testStances) {
			titles.add(s.get(0));
		}
		for (String t : titles) {
			double[] tVec = getGoogleVec(t);
			trainTVecsMap.put(t, tVec);
		}

		trainTVecsMap.save();
		trainTVecsMap.close();

		// getting bodies vectors
		FileHashMap<Integer, Map<Integer, double[]>> trainBVecsMap = new FileHashMap<Integer, Map<Integer, double[]>>(
				bodiesVecPath, FileHashMap.FORCE_OVERWRITE);
		for (Entry<Integer, Map<Integer, String>> e : trainingSummIdBoyMap.entrySet()) {
			String bTxt = "";
			Map<Integer, double[]> partsVecs = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				bTxt = e.getValue().get(i);
				double[] bVec = getGoogleVec(bTxt);
				partsVecs.put(i, bVec);
			}
			trainBVecsMap.put(e.getKey(), partsVecs);

		}

		for (Entry<Integer, Map<Integer, String>> e : testSummIdBoyMap.entrySet()) {
			String bTxt = "";
			Map<Integer, double[]> partsVecs = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				bTxt += e.getValue().get(i);
				double[] bVec = getGoogleVec(bTxt);
				partsVecs.put(i, bVec);
			}
			trainBVecsMap.put(e.getKey(), partsVecs);

		}

		trainBVecsMap.save();
		trainBVecsMap.close();

	}

	private static void generateWord2VecAddSimFeaturesAndSave(List<List<String>> stances, String filename)
			throws IOException {

		FileHashMap<String, double[]> titlesVec = loadTitlesVecs();
		FileHashMap<Integer, Map<Integer, double[]>> bodiesVecs = loadBodiesVecs();

		//
		generateWord2VecSim(stances, filename, titlesVec, bodiesVecs);

	}

	private static void generateWord2VecSim(List<List<String>> trainingStances, String filename,
			FileHashMap<String, double[]> titlesVec, FileHashMap<Integer, Map<Integer, double[]>> bodiesVecs)
			throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "sim" });

		int i = 0;
		for (List<String> stance : trainingStances) {

			List<String> entry = new ArrayList<>();
			entry.add(stance.get(0));
			entry.add(stance.get(1));
			entry.add(stance.get(2));

			double[] tVec = titlesVec.get(stance.get(0));
			Map<Integer, double[]> bVec = bodiesVecs.get(Integer.valueOf(stance.get(1)));

			INDArray tVec_ = Nd4j.create(tVec);
			for (Entry<Integer, double[]> e : bVec.entrySet()) {
				INDArray bVec_ = Nd4j.create(e.getValue());

				double sim = Transforms.cosineSim(tVec_, bVec_);
				entry.add(String.valueOf(sim));
			}

			entries.add(entry.toArray(new String[0]));

			// if(i==10)
			// break;

			if (i % 10000 == 0)
				System.out.println("processed: " + i);
			i++;
		}

		CSVWriter writer = new CSVWriter(new FileWriter(filename));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");
	}

	private static FileHashMap<Integer, Map<Integer, double[]>> loadBodiesVecs() {
		FileHashMap<Integer, Map<Integer, double[]>> bodiesVecs = null;
		try {
			bodiesVecs = new FileHashMap<Integer, Map<Integer, double[]>>(ProjectPaths.BODIES_W2V_SUM2,
					FileHashMap.RECLAIM_FILE_GAPS);
		} catch (ObjectExistsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (VersionMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return bodiesVecs;
	}

	private static FileHashMap<String, double[]> loadTitlesVecs() {
		FileHashMap<String, double[]> titlesVecs = null;
		try {
			titlesVecs = new FileHashMap<String, double[]>("C:/thesis_stuff/help_files/titles_sum_word2vec",
					FileHashMap.RECLAIM_FILE_GAPS);
		} catch (ObjectExistsException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (VersionMismatchException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return titlesVecs;
	}

	private static double[] getGoogleVec(String txt) {
		List<String> lemmas = lemm.lemmatize(txt);
		INDArray identity = Nd4j.zeros(1, 300);

		// System.out.println(identity.shapeInfoToString());

		ArrayList<INDArray> m = new ArrayList<>();
		List<String> gTok = new ArrayList<>();
		for (String tok : lemmas) {
			INDArray tokVec; //
			if (vec.hasWord(tok)) {
				// getting vector for each word method
				// System.out.println("hehe " +
				// Nd4j.create(vec.getWordVector(tok.trim())));
				if (Nd4j.create(vec.getWordVector(tok.trim())) != null)
					tokVec = Nd4j.create(vec.getWordVector(tok.trim()));
				else
					continue;
				// System.out.println(tok);
			} else
				tokVec = identity;
			m.add(tokVec);
		}
		/*
		 * for (String tok : lemmas) { //if
		 * (vec.hasWord(FeatureExtractor.clean(tok).trim())) { //Crabzilla
		 * problem if(!FeatureExtractor.clean(tok).trim().isEmpty() &&
		 * vec.hasWord(FeatureExtractor.clean(tok).trim())){ gTok.add(tok); } }
		 */
		/*
		 * INDArray result;
		 * 
		 * if(gTok.size() != 0){ result = vec.getWordVectorsMean(gTok); }else
		 * result = Nd4j.zeros(1, 300);
		 */

		INDArray result = m.stream().reduce(identity, INDArray::add);
		double[] arrResult = getArrayVec(result);
		if (arrResult == null)
			System.out.println(lemmas);
		return arrResult;
	}

	private static double[] getArrayVec(INDArray result) {
		double[] arrResult = result.data().asDouble();

		return arrResult;
	}

	private static Word2Vec loadGoogleNewsVec() {
		return WordVectorSerializer.readWord2VecModel("C:/thesis_stuff/GoogleNews-vectors-negative300.bin.gz", false);
	}
	
	public static void saveWord2VecFeaturesAsHashFile(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, double[]> w2vSimData = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.FORCE_OVERWRITE);

		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			double[] f = new double[3];
			for(int i = 0; i < 3; i++)
				f[i] = Double.valueOf(line[i + 3]);
			w2vSimData.put(line[0] + line[1], f);
		}
		reader.close();

		// saving the map file
		w2vSimData.save();
		w2vSimData.close();
	}
	
	public static FileHashMap<String, double[]> loadWord2VecFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, double[]> w2vSimData = new FileHashMap<String, double[]>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return w2vSimData;
	}

}
