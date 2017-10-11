package ude.master.thesis.stance_detection.processor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import edu.stanford.nlp.ie.util.RelationTriple;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.naturalli.NaturalLogicAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;
import ude.master.thesis.stance_detection.util.PPDBProcessor;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class SVOFeaturesGeneratorWithArguments {

	// private static BufferedWriter out;

	private static Map<String, Integer> entailmentsMap;
	private static Map<Integer, String> trainIdBodyMap;
	private static List<List<String>> trainingStances;
	private static HashMap<Integer, String> testIdBodyMap;
	private static List<List<String>> testStances;
	private static FileHashMap<String, List<Map<String, String>>> titlesSVOsMap;
	private static FileHashMap<String, List<Map<String, String>>> bodiesSVOsMap;
	private static FileHashMap<String, ArrayList<ArrayList<String>>> ppdbData;
	private static Map<Integer, String> trainingIdBoyMap;
	private static HashMap<Integer, String> testIdBoyMap;

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		/*
		 * StanceDetectionDataReader sddr = new StanceDetectionDataReader(true,
		 * true, "resources/data/train_stances.csv",
		 * "resources/data/summ_train_bodies.csv",
		 * "resources/data/test_data/competition_test_stances.csv",
		 * "resources/data/test_data/summ_competition_test_bodies.csv");
		 * 
		 * Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		 * List<List<String>> trainingStances = sddr.getTrainStances();
		 * HashMap<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		 * List<List<String>> testStances = sddr.getTestStances();
		 * 
		 * String headline = "ISIL Beheads American Photojournalist in Iraq";
		 * String body =
		 * "James Foley, an American journalist who went missing in Syria more than a year ago, "
		 * +
		 * "has reportedly been executed by the Islamic State, a militant group formerly known as ISIS. "
		 * +
		 * "Video and photos purportedly of Foley emerged on Tuesday. A YouTube video -- entitled \"\"A "
		 * +
		 * "Message to #America (from the #IslamicState)\"\" -- identified a man on his knees as"
		 * +
		 * " \"\"James Wright Foley,\"\" and showed his execution. This is a developing story. "
		 * + "Check back here for updates.";
		 * 
		 * String headline1 =
		 * "Banksy 'arrested & real identity revealed' is the same hoax from last year"
		 * ; String headline3 =
		 * "BrFidel Castro is dead, according to viral Twitter rumors"; String
		 * testTxt = "We have no information on whether users are at risk";
		 */

		StanceDetectionDataReader sddr = null;
		try {
			sddr = new StanceDetectionDataReader(true, true, ProjectPaths.TRAIN_STANCES_PREPROCESSED,
					ProjectPaths.ARGUMENTED_BODIES_TRAIN, ProjectPaths.TEST_STANCESS_PREPROCESSED,
					ProjectPaths.ARGUMENTED_BODIES_TEST);

		} catch (IOException e) { // TODO Auto-generated catch block
			e.printStackTrace();
		}

		trainingIdBoyMap = sddr.getTrainIdBodyMap();
		testIdBoyMap = sddr.getTestIdBodyMap();

		trainingStances = sddr.getTrainStances(); 
		testStances = sddr.getTestStances();

		entailmentsMap = new HashMap<>();
		entailmentsMap.put("ReverseEntailment", 0);
		entailmentsMap.put("ForwardEntailment", 1);
		entailmentsMap.put("Equivalence", 2);
		entailmentsMap.put("OtherRelated", 2);
		entailmentsMap.put("Independence", 3);

		//extractTitlesAndBodiesSVOsAndSave();
		// ppdb: paraphrase, score, entailment
		ppdbData = PPDBProcessor.loadPPDB2(PPDBProcessor.MAP_PPDB_2_XXL_ALL);

		try {
			titlesSVOsMap = new FileHashMap<String, List<Map<String, String>>>(ProjectPaths.TITLES_SVO_TRAIN_TEST,
					FileHashMap.RECLAIM_FILE_GAPS);
			bodiesSVOsMap = new FileHashMap<String,List<Map<String, String>>>(ProjectPaths.BODIES_SVO_ARG_TRAIN_TEST, FileHashMap.RECLAIM_FILE_GAPS);

			generateDataSummedSVOFeatureVector(trainingStances, ProjectPaths.CSV_SUMMED_SVO_FEATURE_ARG_TRAIN);
			generateDataSummedSVOFeatureVector(testStances, ProjectPaths.CSV_SUMMED_SVO_FEATURE_ARG_TEST);
		} catch (ObjectExistsException | ClassNotFoundException | VersionMismatchException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		 saveSVOFeaturesAsHashFile(ProjectPaths.CSV_SUMMED_SVO_FEATURE_ARG_TRAIN, ProjectPaths.SUMMED_SVO_FEATURE_ARG_TRAIN);
		 saveSVOFeaturesAsHashFile(ProjectPaths.CSV_SUMMED_SVO_FEATURE_ARG_TEST, ProjectPaths.SUMMED_SVO_FEATURE_ARG_TEST);
	}


	public static void generateDataSummedSVOFeatureVector(List<List<String>> stances, String csvFilepath)
			throws Exception {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "svo_feature" });

		int k = 0;
		// get svos for training data
		for (List<String> s : stances) {

			List<String> entry = new ArrayList<>();
			entry.add(s.get(0));
			entry.add(s.get(1));
			entry.add(s.get(2));

			// int vec[] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

			List<Map<String, String>> t_svos = null;

			t_svos = titlesSVOsMap.get(s.get(0));

			List<Map<String, String>>  b_svos = bodiesSVOsMap.get(s.get(1));

			List<int[]> vecs = new ArrayList<>();
			int vec[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

				if (!t_svos.isEmpty() && !b_svos.isEmpty()) {
					// Map<int[], Double> vecMap = new HashMap<>();

					for (Map<String, String> t_svo : t_svos) {
						for (Map<String, String> b_svo : b_svos) {
							
							int[] nsubjEntailment = calcEntailmentFeatureVector(t_svo.get("nsubj"), b_svo.get("nsubj"));
							int[] verbEntailment = calcEntailmentFeatureVector(t_svo.get("verb"), b_svo.get("verb"));
							int[] dobjEntailment = calcEntailmentFeatureVector(t_svo.get("dobj"), b_svo.get("dobj"));

							int v[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
							for (int i = 0; i < 4; i++) {
								v[i] = nsubjEntailment[i];
							}

							for (int i = 4; i < 8; i++) {
								v[i] = verbEntailment[i - 4];
							}

							for (int i = 8; i < 12; i++) {
								v[i] = dobjEntailment[i - 8];
							}
							vecs.add(v);

						}
					}

				} else {
					int v[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
					vecs.add(v);
				}

			
			// add up the svo vectors
			int[] identity = new int[12];
			Arrays.setAll(identity, (index) -> 0);
			int[] vv = vecs.stream().reduce(identity, SVOFeaturesGeneratorWithArguments::add);

			vec = Arrays.copyOf(vv, vec.length);

			entry.add(Arrays.toString(vec));
			// System.out.println(features.size() + " " + entry.toString());
			if (vec.length != 12)
				throw new Exception(" not 12");

			entries.add(entry.toArray(new String[0]));

			if (k % 10000 == 0)
				System.out.println("processed: " + k);
			k++;

		}

		CSVWriter writer = new CSVWriter(new FileWriter(csvFilepath));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");
	}

	public static int[] add(int[] first, int[] second) {
		int length = first.length < second.length ? first.length : second.length;
		int[] result = new int[length];

		for (int i = 0; i < length; i++) {
			result[i] = first[i] + second[i];
		}

		return result;
	}

	private static Double avg(int[] v) {
		double sum = 0.0;
		for (int i = 0; i < v.length; i++)
			sum += v[i];
		return sum / v.length;
	}

	public static int[] calcEntailmentFeatureVector(String w1, String w2) throws FileNotFoundException, IOException {
		int vec[] = { 0, 0, 0, 0 };

		if (w1.equals(w2)) {
			vec[entailmentsMap.get("Equivalence")] = 1;
		}

		ArrayList<ArrayList<String>> w1Paraphrases = ppdbData.get(w1);
		if (w1Paraphrases != null) {
			List<Integer> relationships = new ArrayList<>();
			for (ArrayList<String> p : w1Paraphrases) {
				if (p.get(0).equals(w2)) {
					if (entailmentsMap.containsKey(p.get(2)))
						relationships.add(entailmentsMap.get(p.get(2)));
				}
			}
			if (!relationships.isEmpty()) {
				Integer relationship = Collections.max(relationships);
				vec[relationship] = 1;
			}
		}
		for(int v : vec)
			if(v !=0 )
				System.out.println(Arrays.toString(vec));
		return vec;

	}

	public static void extractTitlesAndBodiesSVOsAndSave() throws Exception {

		StanfordCoreNLP pipeline = getStanfordPipeline();

		// get titles SVOs
/*		String titlesSVOsPath = ProjectPaths.TITLES_SVO_TRAIN_TEST;
		FileHashMap<String, List<Map<String, String>>> titlesSVOs = new FileHashMap<String, List<Map<String, String>>>(
				titlesSVOsPath, FileHashMap.FORCE_OVERWRITE);

		Set<String> titles = new HashSet<>();
		for (List<String> s : trainingStances) {
			titles.add(s.get(0));
		}
		for (List<String> s : testStances) {
			titles.add(s.get(0));
		}
		for (String t : titles) {
			List<Map<String, String>> svos = getSVOsFromText(pipeline, t);
			titlesSVOs.put(t, svos);
		}

		titlesSVOs.save();
		titlesSVOs.close();
*/
		// get bodies SVOs

		String bodiesSVOsPath = ProjectPaths.BODIES_SVO_ARG_TRAIN_TEST;
		FileHashMap<String, List<Map<String, String>>> bodiesSVOs = new FileHashMap<String, List<Map<String, String>>>(
				bodiesSVOsPath, FileHashMap.FORCE_OVERWRITE);

		for (Entry<Integer, String> e : trainingIdBoyMap.entrySet()) {
				String body = e.getValue();
				// if (!part.isEmpty()) {
				List<Map<String, String>> svos = getSVOsFromText(pipeline, body);
				
				// }
			bodiesSVOs.put(String.valueOf(e.getKey()), svos);
		}
		for (Entry<Integer, String> e : testIdBoyMap.entrySet()) {
				String body = e.getValue();
				// if (!part.isEmpty()) {
				List<Map<String, String>> svos = getSVOsFromText(pipeline, body);


			bodiesSVOs.put(String.valueOf(e.getKey()), svos);
		}

		bodiesSVOs.save();
		bodiesSVOs.close();

	}

	private static List<Map<String, String>> getSVOsFromText(StanfordCoreNLP pipeline, String t) {
		// System.out.println("text = " + t);
		Annotation doc = new Annotation(t);
		pipeline.annotate(doc);
		List<CoreMap> sentences = doc.get(SentencesAnnotation.class);

		List<SemanticGraph> graphs = StanfordDependencyParser.buildDependencyGraph(doc);

		// List of svos in t
		List<Map<String, String>> svos = new ArrayList<>();

		int i = 0;
		for (CoreMap sentence : sentences) {
			Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);

			SemanticGraph relatedGraph = graphs.get(i);
			// System.out.println("triples.size = "+triples.size());
			// int j =0;
			for (RelationTriple triple : triples) {
				// System.out.println("triple = " +
				// triple.toQaSrlString(sentence));
				List<CoreLabel> tokens = triple.allTokens();

				Map<String, String> vec = new HashMap<String, String>();
				for (CoreLabel tok : tokens) {

					int tIndex = tok.index();

					Set<GrammaticalRelation> relns = relatedGraph.relns(relatedGraph.getNodeByIndexSafe(tIndex));
					for (GrammaticalRelation rel : relns) {
						if (rel.toString().equals("nsubj")) {
							vec.put("nsubj", tok.lemma());
						}
						if (rel.toString().equals("dobj")) {
							vec.put("dobj", tok.lemma());
						}
					}
				}

				if (vec.size() == 2) {
					// find the verb
					String depList = relatedGraph.toList();
					String[] deps = depList.split("\n");
					for (String d : deps) {
						String depType = d.substring(0, d.indexOf('('));
						// System.out.println(d + " "+sentence.toString());
						if (depType.equals("nsubj")) {
							String betweenBrack = d.substring(d.lastIndexOf('(') + 1, d.indexOf(')'));
							String[] depWords = betweenBrack.split(",");
							// System.out.println(depWords[0].substring(depWords[0].lastIndexOf('-')
							// + 1));
							if ((depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim()).contains("'''")) {
								System.out.println("haha ");
								int verbIdx = Integer.valueOf(depWords[0]
										.substring(depWords[0].lastIndexOf('-') + 1, depWords[0].indexOf("'")).trim());
								System.out.print(relatedGraph.getNodeByIndex(verbIdx).lemma());
								vec.put("verb", relatedGraph.getNodeByIndex(verbIdx).lemma());
								continue;
							}
							int verbIdx = Integer
									.valueOf(depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim());
							vec.put("verb", relatedGraph.getNodeByIndex(verbIdx).lemma());

						}
					}
				}

				if (vec.size() == 3) {
					// System.out.println(vec);
					if (!svos.toString().contains(vec.toString()))
						svos.add(vec);
				}

			}
			i++;
		}
		// System.out.println("svos = " + svos);
		return svos;

	}

	/**
	 * 
	 * @param hashFileName
	 * @throws IOException
	 * @throws VersionMismatchException
	 * @throws ClassNotFoundException
	 * @throws ObjectExistsException
	 * @throws FileNotFoundException
	 */
	public static void saveTitlesDependencyGraphs(String hashFileName) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		// saving titles graphs
		FileHashMap<String, List<Map<String, String>>> titlesDepTreeMap = new FileHashMap<String, List<Map<String, String>>>(
				hashFileName + "_titles", FileHashMap.FORCE_OVERWRITE);

		Set<String> titles = new HashSet<String>();

		for (List<String> s : trainingStances) {
			titles.add(s.get(0));
		}

		for (List<String> s : testStances) {
			titles.add(s.get(0));
		}

		StanfordCoreNLP pipeline = getStanfordPipeline();

		for (String t : titles) {
			List<Map<String, String>> svos = getSVOsFromText(pipeline, t);
			titlesDepTreeMap.put(t, svos);
		}

		titlesDepTreeMap.save();
		titlesDepTreeMap.close();

		// saving bodies graphs
		FileHashMap<String, List<Map<String, String>>> bodiesDepTreeMap = new FileHashMap<String, List<Map<String, String>>>(
				hashFileName + "_bodies", FileHashMap.FORCE_OVERWRITE);
		for (Map.Entry<Integer, String> b : trainIdBodyMap.entrySet()) {
			List<Map<String, String>> svos = getSVOsFromText(pipeline, b.getValue());
			bodiesDepTreeMap.put(String.valueOf(b.getKey()), svos);
		}

		for (Map.Entry<Integer, String> b : testIdBodyMap.entrySet()) {
			List<Map<String, String>> svos = getSVOsFromText(pipeline, b.getValue());
			bodiesDepTreeMap.put(String.valueOf(b.getKey()), svos);

		}

		bodiesDepTreeMap.save();
		bodiesDepTreeMap.close();

	}

	private static StanfordCoreNLP getStanfordPipeline() {
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse,natlog,openie");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		return pipeline;
	}

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

	public static FileHashMap<String, ArrayList<Integer>> loadSVOFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, ArrayList<Integer>> svoData = new FileHashMap<String, ArrayList<Integer>>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return svoData;
	}

	public static void saveSVOSummedFeaturesAsHashFile(String csvfilePath, String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
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
}
