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
import java.util.Properties;
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

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class Word2VecDataGenerator33MidArguments {

	private static Word2Vec vec;
	private static Lemmatizer lemm;

	private static StanfordCoreNLP pipeline;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {

		//vec = loadGoogleNewsVec();
		//lemm = new Lemmatizer();

		//generateWord2VecData();
		saveWord2VecFeaturesAsHashFile(ProjectPaths.CSV_TRAIN_PARTS_ARG33_W2V_SUM_SIM_PARTS,
				ProjectPaths.TRAIN_W2V_SUM_SIM_PARTS_ARG33);
		saveWord2VecFeaturesAsHashFile(ProjectPaths.CSV_TEST_PARTS_ARG33_W2V_SUM_SIM_PARTS,
				ProjectPaths.TEST_W2V_SUM_SIM_PARTS_ARG33);
	}

	private static void generateWord2VecData()
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES33_TEST);

		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TRAIN));
		List<List<String>> trainingStances = sddr.getTrainStances();

		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES33_TEST));
		List<List<String>> testStances = sddr.getTestStances();

		generateAddWord2VecAndSave(trainingSummIdBoyMap, trainingStances, testSummIdBoyMap, testStances,
				ProjectPaths.BODIES_PARTS_ARG33_W2V_SUM, ProjectPaths.TITLES_PARTS_ARG33_W2V_SUM);

		generateWord2VecAddSimFeaturesAndSave(trainingStances, ProjectPaths.CSV_TRAIN_PARTS_ARG33_W2V_SUM_SIM_PARTS);
		generateWord2VecAddSimFeaturesAndSave(testStances, ProjectPaths.CSV_TEST_PARTS_ARG33_W2V_SUM_SIM_PARTS);

	}

	private static void generateAddWord2VecAndSave(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			List<List<String>> trainingStances, HashMap<Integer, Map<Integer, String>> testSummIdBoyMap,
			List<List<String>> testStances, String bodiesVecPath, String titlesVecPath) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		// getting titles vectors
		FileHashMap<String, Double[]> titlesVecsMap = new FileHashMap<String, Double[]>(titlesVecPath,
				FileHashMap.FORCE_OVERWRITE);
		Set<String> titles = new HashSet<>();

		for (List<String> s : trainingStances) {
			titles.add(s.get(0));

		}
		for (List<String> s : testStances) {
			titles.add(s.get(0));
		}
		for (String t : titles) {
			Double[] tVec = getGoogleVecFromSentence(t);
			titlesVecsMap.put(t, tVec);
		}

		titlesVecsMap.save();
		titlesVecsMap.close();

		// getting bodies vectors
		FileHashMap<Integer, Map<Integer, List<Double[]>>> bodiesVecsMap = new FileHashMap<Integer, Map<Integer, List<Double[]>>>(
				bodiesVecPath, FileHashMap.FORCE_OVERWRITE);

		// process training bodies
		for (Entry<Integer, Map<Integer, String>> e : trainingSummIdBoyMap.entrySet()) {
			String bTxt = "";
			Map<Integer, List<Double[]>> partsVecs = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				bTxt = e.getValue().get(i);
				List<Double[]> bVec = getGoogleVecsFromParagraph(bTxt);
				partsVecs.put(i, bVec);
			}
			bodiesVecsMap.put(e.getKey(), partsVecs);

		}

		// process test bodies
		for (Entry<Integer, Map<Integer, String>> e : testSummIdBoyMap.entrySet()) {
			String bTxt = "";
			Map<Integer, List<Double[]>> partsVecs = new HashMap<>();
			for (int i = 1; i <= 3; i++) {
				bTxt = e.getValue().get(i);
				List<Double[]> bVec = getGoogleVecsFromParagraph(bTxt);
				partsVecs.put(i, bVec);
			}
			bodiesVecsMap.put(e.getKey(), partsVecs);

		}

		bodiesVecsMap.save();
		bodiesVecsMap.close();

	}

	private static StanfordCoreNLP initStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

	private static List<Double[]> getGoogleVecsFromParagraph(String bTxt) {

		if (pipeline == null)
			pipeline = initStanfordPipeline();

		Annotation doc = new Annotation(bTxt);
		pipeline.annotate(doc);

		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		List<Double[]> pvecs = new ArrayList<>();
		for (CoreMap s : sentences) {
			Double[] svec = getGoogleVecFromSentence(s.toString());
			pvecs.add(svec);
		}
		return pvecs;
	}

	private static Double[] getGoogleVecFromSentence(String txt) {
		List<String> lemmas = lemm.lemmatize(txt);
		INDArray identity = Nd4j.zeros(1, 300);

		// System.out.println(identity.shapeInfoToString());

		ArrayList<INDArray> m = new ArrayList<>();
		for (String tok : lemmas) {
			INDArray tokVec; //
			if (vec.hasWord(tok)) {
				if (Nd4j.create(vec.getWordVector(tok.trim())) != null)
					tokVec = Nd4j.create(vec.getWordVector(tok.trim()));
				else
					continue;
				// System.out.println(tok);
			} else
				tokVec = identity;
			m.add(tokVec);
		}

		INDArray result = m.stream().reduce(identity, INDArray::add);
		Double[] arrResult = getArrayVec(result);
		if (arrResult == null)
			System.out.println(lemmas);
		return arrResult;
	}

	private static void generateWord2VecAddSimFeaturesAndSave(List<List<String>> stances, String filename)
			throws IOException {

		FileHashMap<String, Double[]> titlesVec = loadTitlesVecs(ProjectPaths.TITLES_PARTS_ARG33_W2V_SUM);
		FileHashMap<Integer, Map<Integer, List<Double[]>>> bodiesVecs = loadBodiesVecs(
				ProjectPaths.BODIES_PARTS_ARG33_W2V_SUM);

		//
		generateWord2VecSim(stances, filename, titlesVec, bodiesVecs);

	}

	private static void generateWord2VecSim(List<List<String>> trainingStances, String filename,
			FileHashMap<String, Double[]> titlesVec, FileHashMap<Integer, Map<Integer, List<Double[]>>> bodiesVecs)
			throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "sim_0", "sim_1", "sim_2", "sim_3", "sim_4", "sim_5",
				"sim_left_sent" });

		int i = 0;
		for (List<String> stance : trainingStances) {

			List<String> entry = new ArrayList<>();
			entry.add(stance.get(0));
			entry.add(stance.get(1));
			entry.add(stance.get(2));
			if (bodiesVecs.containsKey(Integer.valueOf(stance.get(1)))) {
				Double[] tVec = titlesVec.get(stance.get(0));
				Map<Integer, List<Double[]>> bVec = bodiesVecs.get(Integer.valueOf(stance.get(1)));

				double[] tVecd = new double[tVec.length];
				for (int h = 0; h < tVec.length; h++)
					tVecd[h] = tVec[h];

				INDArray tVec_ = Nd4j.create(tVecd);
				for (Entry<Integer, List<Double[]>> e : bVec.entrySet()) {
					if (e.getKey() == 1 || e.getKey() == 3)
						for (Double[] v : e.getValue()) {

							double[] vd = new double[v.length];
							for (int h = 0; h < v.length; h++)
								vd[h] = v[h];

							INDArray bVec_ = Nd4j.create(vd);

							double sim = Transforms.cosineSim(tVec_, bVec_);
							if (String.valueOf(sim).equals("NaN")) {
								entry.add(String.valueOf(0.0));
							} else
								entry.add(String.valueOf(sim));
						}
				}

				for (Entry<Integer, List<Double[]>> e : bVec.entrySet()) {
					if (e.getKey() == 2)
						for (Double[] v : e.getValue()) {
							
							double[] vd = new double[v.length];
							for (int h = 0; h < v.length; h++)
								vd[h] = v[h];

							INDArray bVec_ = Nd4j.create(vd);

							double sim = Transforms.cosineSim(tVec_, bVec_);

							if (String.valueOf(sim).equals("NaN")) {
								entry.add(String.valueOf(0.0));
							} else
								entry.add(String.valueOf(sim));
							//entry.add(String.valueOf(sim));
						}
				}
				entries.add(entry.toArray(new String[0]));

			}
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

	private static FileHashMap<Integer, Map<Integer, List<Double[]>>> loadBodiesVecs(String bodiesVecPath) {
		FileHashMap<Integer, Map<Integer, List<Double[]>>> bodiesVecs = null;
		try {
			bodiesVecs = new FileHashMap<Integer, Map<Integer, List<Double[]>>>(bodiesVecPath,
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

	private static FileHashMap<String, Double[]> loadTitlesVecs(String titlesVecPath) {
		FileHashMap<String, Double[]> titlesVecs = null;
		try {
			titlesVecs = new FileHashMap<String, Double[]>(titlesVecPath, FileHashMap.RECLAIM_FILE_GAPS);
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

	private static Double[] getArrayVec(INDArray result) {
		double[] arrResult = result.data().asDouble();

		Double[] arrResultD = new Double[arrResult.length];
		int i = 0;
		for (double d : arrResult) {
			arrResultD[i] = d;
			i++;
		}
		return arrResultD;
	}

	private static Word2Vec loadGoogleNewsVec() {
		return WordVectorSerializer.readWord2VecModel("C:/thesis_stuff/GoogleNews-vectors-negative300.bin.gz", false);
	}

	/**
	 * Save the w2v cos similarity for the first 3 sentences and then for the
	 * last 3 and then for the first 2 sentences from the middle part (using
	 * -100.0 if there is not enough sent.) --> 8 features values
	 * 
	 * @param csvfilePath
	 * @param hashFileName
	 * @throws FileNotFoundException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 * @throws IOException
	 */
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
			double[] f = new double[8];
			for (int i = 0; i < 6; i++)
				f[i] = Double.valueOf(line[i + 3]);

			for (int i = 6; i < 8; i++)
				if (i < line.length - 3)
					f[i] = Double.valueOf(line[i + 3]);
				else
					f[i] = Double.valueOf(-100.0);

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
