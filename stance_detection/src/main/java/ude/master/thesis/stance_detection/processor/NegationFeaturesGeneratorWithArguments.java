package ude.master.thesis.stance_detection.processor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import java_cup.production;
import ude.master.thesis.stance_detection.util.BodySummerizer2;
import ude.master.thesis.stance_detection.util.PPDBProcessor;
import ude.master.thesis.stance_detection.util.PPDBProcessorWithArguments;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;
import ude.master.thesis.stance_detection.util.TitleAndBodyTextPreprocess;

public class NegationFeaturesGeneratorWithArguments {

	private static Lemmatizer lemm;
	private static Porter porter;
	private static BufferedWriter out;
	private static FileHashMap<Integer, List<String>> trainBMap;
	private static FileHashMap<String, List<String>> trainTMap;
	private static FileHashMap<Integer, List<String>> testBMap;
	private static FileHashMap<String, List<String>> testTMap;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		/*
		 * String testTxt =
		 * "Newly released audio allegedly records the moment that Officer Darren Wilson "
		 * +
		 * "opened fire on unarmed Michael Brown At least ten shots can be heard - in two separate "
		 * +
		 * "volleys of gunfire Experts have said this indicated a 'moment of contemplation' for Wilson "
		 * + "FBI has confirmed it has interviewed the man who recorded audio "
		 * +
		 * "Is another tantalizing piece of evidence collected in the ongoing case Officer Wilson "
		 * +
		 * "claims he felt his life was threatened on August 9 Witnesses and a "
		 * +
		 * "friend of Brown, 18, claim he had surrendered Brown was buried on Monday in a ceremony "
		 * +
		 * "attended by thousands The FBI has been handed a potentially crucial recording that allegedly "
		 * +
		 * "contains audio of the moment that Officer Darren Wilson opened fire and killed unarmed "
		 * +
		 * "18-year-old Michael Brown in Ferguson, Missouri, earlier this month. Since the guard's "
		 * +
		 * "arrival Monday, flare-ups in the small section of town that had been the center of nightly "
		 * +
		 * "unrest have begun to subside. About 100 people gathered Thursday evening, walking in laps "
		 * +
		 * "near the spot where Michael Brown was shot. Some were in organized groups, such as clergy "
		 * +
		 * "members. More signs reflected calls by protesters to remove the prosecutor from the case."
		 * ;
		 * 
		 * // Body ID 1254 String testTxt1 =
		 * "The Islamic State (IS) leader Abu Bakr al-Baghdadi has not been killed as has "
		 * +
		 * "been previously claimed. He is wounded and being treated in the border area of Iraq and "
		 * +
		 * "Syria. A few days ago, it was reported that al-Baghdadi had been killed by a U.S. "
		 * +
		 * "airstrike near Mosul in Northern Iraq, an attack that left three other senior members of "
		 * +
		 * "the militant group dead. When it was reported that al-Baghdadi had been killed, the Pentagon "
		 * +
		 * "did not confirm the death, but thousands of social media users shared an unverified photo "
		 * +
		 * "claiming to be the ISIS leader’s body. However, Pentagon spokesman Col. Steve Warren did "
		 * +
		 * "later say that any IS leaders “inside troop formations are likely to be killed.”"
		 * ;
		 * 
		 * String file;
		 */
		// List<Integer> negIdxs = getNegationIdxs(testTxt1);
		// testing stuff
		/**
		 * System.out.println(lemm.lemmatize(testTxt1));
		 * 
		 * String cleanH = FeatureExtractor.clean(testTxt1); List
		 * <String> hLemmas = lemm.lemmatize(cleanH); hLemmas =
		 * FeatureExtractor.removeStopWords(hLemmas);
		 * 
		 * System.out.println(hLemmas);
		 * 
		 * 
		 * 
		 * System.out.println(lemmaMap);
		 * 
		 * System.out.println(lemmaMap.size() + "  " + hLemmas.size());
		 */
		// out = new BufferedWriter(new
		// FileWriter("resources/words_not_processed.txt"));
		// out.write("==========================");
		// out.newLine();

		lemm = new Lemmatizer();
		porter = new Porter();

		getNegFeaturesForData();

		saveNegFeaturesAsHashFile(ProjectPaths.CSV_NEG_FEATURE_ARG_TRAIN, ProjectPaths.NEG_FEATURE_ARG_TRAIN);
		saveNegFeaturesAsHashFile(ProjectPaths.CSV_NEG_FEATURE_ARG_TEST, ProjectPaths.NEG_FEATURE_ARG_TEST);

		// out.flush();
		// out.close();
		/*
		 * lemm = new Lemmatizer(); String txt =
		 * "Kim Jong-un has broken both of his ankles and is now in the hospital after undergoing "
		 * +
		 * "surgery, a report in a South Korean newspaper claims. The North Korean leader has "
		 * +
		 * "been missing for more than three weeks, fueling speculation about what could cause his "
		 * +
		 * "unusual disappearance from the public eye. This rumor seems to confirm what North Korean "
		 * +
		 * "state media had said on Thursday, when state broadcaster Korean Central Television "
		 * +
		 * "reported that Kim was \"not feeling well,\" and was suffering from an \"uncomfortable "
		 * +
		 * "physical condition.\" Have something to add to this story? Share it in the comments."
		 * ; System.out.println(lemm.lemmatize(txt));
		 * 
		 * System.out.println(lemm.lemmatize("I'm not feeling well"));
		 */

		/*
		 * trainBMap =
		 * getBodiesDepMap("C:/thesis_stuff/help_files/train_bodies_deps");
		 * trainTMap =
		 * getTitlesDepMap("C:/thesis_stuff/help_files/train_titles_deps");
		 * 
		 * testBMap =
		 * getBodiesDepMap("C:/thesis_stuff/help_files/test_bodies_deps");
		 * testTMap =
		 * getTitlesDepMap("C:/thesis_stuff/help_files/test_titles_deps");
		 * 
		 * System.out.println(trainBMap.get(1144));
		 */
	}

	private static void getNegFeaturesForData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_BODIES_TEST);

		trainBMap = loadBodiesDepMap(ProjectPaths.TRAIN_ARG_BODIES_DEPS);
		trainTMap = loadTitlesDepMap(ProjectPaths.TRAIN_TITLES_DEPS2); // The
																		// same
																		// in
																		// the
																		// arguments
																		// experience

		testBMap = loadBodiesDepMap(ProjectPaths.TEST_ARG_BODIES_DEPS);
		testTMap = loadTitlesDepMap(ProjectPaths.TEST_TITLES_DEPS2);// The same
																	// in the
																	// arguments
																	// experience

		System.out.println("trainTMap.size" + trainTMap.size());
		Map<Integer, String> trainingIdBoyMap = sddr.getTrainIdBodyMap();
		List<List<String>> trainingStances = sddr.getTrainStances();

		generateNegFeaturesAndSave(trainingIdBoyMap, trainingStances, trainTMap, trainBMap,
				ProjectPaths.CSV_NEG_FEATURE_ARG_TRAIN, ProjectPaths.SPPDB_HUNG_SCORES_IDXS_ARG_TRAIN);

		Map<Integer, String> testIdBoyMap = sddr.getTestIdBodyMap();
		List<List<String>> testStances = sddr.getTestStances();
		generateNegFeaturesAndSave(testIdBoyMap, testStances, testTMap, testBMap,
				ProjectPaths.CSV_NEG_FEATURE_ARG_TEST, ProjectPaths.SPPDB_HUNG_SCORES_IDXS_ARG_TEST);

	}

	private static FileHashMap<String, ArrayList<ArrayList<Integer>>> loadHungarianScores(String hungarianScorePath) {
		FileHashMap<String, ArrayList<ArrayList<Integer>>> hung_scores = PPDBProcessorWithArguments
				.loadHungarianScoreFromFileMap(hungarianScorePath);
		return hung_scores;
	}

	private static FileHashMap<Integer, List<String>> loadBodiesDepMap(String path) {
		FileHashMap<Integer, List<String>> bodiesDeps = null;
		try {
			bodiesDeps = new FileHashMap<Integer, List<String>>(path, FileHashMap.RECLAIM_FILE_GAPS);
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
		return bodiesDeps;
	}

	private static FileHashMap<String, List<String>> loadTitlesDepMap(String path) {
		FileHashMap<String, List<String>> titlesDeps = null;
		try {
			titlesDeps = new FileHashMap<String, List<String>>(path, FileHashMap.RECLAIM_FILE_GAPS);
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
		return titlesDeps;
	}

	/**
	 * 
	 * @param trainingIdBoyMap
	 * @param stances
	 * @param featuresFilename
	 *            file name to save the features to
	 * @throws IOException
	 */
	private static void generateNegFeaturesAndSave(Map<Integer, String> trainingIdBoyMap, List<List<String>> stances,
			FileHashMap<String, List<String>> titleDepMap, FileHashMap<Integer, List<String>> trainBMap2,
			String featuresFilename, String hungarianScorePath) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "neg_feature" });

		FileHashMap<String, ArrayList<ArrayList<Integer>>> hungScores = loadHungarianScores(hungarianScorePath);
		int i = 0;
		int negFeature = 0;
		for (List<String> stance : stances) {
			negFeature = 0;
			
			List<String> entry = new ArrayList<>();
			entry.add(stance.get(0));
			entry.add(stance.get(1));
			entry.add(stance.get(2));

			// System.out.println(stance.get(1));
			List<Integer> titleNegIdxs = getTitleNegationIdxs(stance.get(0), titleDepMap);

			// the bodyNegIdxs size should be equal to the no of body sentences
			ArrayList<ArrayList<Integer>> bodyNegIdxs = getBodyNegationIdxs(stance.get(1), trainBMap2);

			
			for (int j = 0; j < bodyNegIdxs.size(); j++) {
				if ((titleNegIdxs.size() == 0) && (bodyNegIdxs.get(j).size() == 0)) {
					//entry.add(String.valueOf(negFeature));
					//When nothing in title or in a body sentence is negated, skip to the next sentence
					continue;
				}

				ArrayList<ArrayList<Integer>> idxsMap = hungScores.get(stance.get(0) + stance.get(1));

				ArrayList<Integer> hScoreidxs = new ArrayList<>();
				hScoreidxs = idxsMap.get(j);


				// System.out.println("idxs = " + idxs);
				// System.out.println("titleNegIdxs = " + titleNegIdxs);
				// System.out.println("bodyNegIdxs" + bodyNegIdxs);

				int tIdx = 0;

				for (tIdx = 0; tIdx < hScoreidxs.size(); tIdx++)
					if ((titleNegIdxs.contains(tIdx + 1) && !bodyNegIdxs.get(j).contains(hScoreidxs.get(tIdx) + 1))
							|| (!titleNegIdxs.contains(tIdx + 1)
									&& bodyNegIdxs.get(j).contains(hScoreidxs.get(tIdx) + 1)))
						negFeature++;

			}
			entry.add(String.valueOf(negFeature));
			entries.add(entry.toArray(new String[0]));

			if (i % 10000 == 0)
				System.out.println("processed: " + i);
			i++;
		}

		CSVWriter writer = new CSVWriter(new FileWriter(featuresFilename));
		writer.writeAll(entries);
		writer.flush();
		writer.close();
		System.out.println("saved saved saved");

	}

	private static List<Integer> getTitleNegationIdxs(String title, FileHashMap<String, List<String>> titleDepMap)
			throws IOException {

		List<String> graphs = null;

		graphs = titleDepMap.get(title);

		List<Integer> negIdxs = new ArrayList<>();
		int i = 0; // Sentence / Graph Index
		for (String depList : graphs) {
			String[] deps = depList.split("\n");
			for (String d : deps) {
				String depType = d.substring(0, d.indexOf('('));
				// System.out.println("depType = " + depType);
				if (depType.equals("neg")) {
					String betweenBrack = d.substring(d.indexOf('(') + 1, d.indexOf(')'));
					String[] depWords = betweenBrack.split(", ");

					String negWord = depWords[0].substring(0, depWords[0].lastIndexOf('-'));
					int negWordIndex = Integer.valueOf(depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim());

					negIdxs.add(negWordIndex);
				}
			}
			i++;
		}

		return negIdxs;
	}

	/**
	 * Retains a list of lists; a list for each sentence containing the indecis
	 * @param bodyId
	 * @param trainBMap
	 * @return
	 * @throws IOException
	 */
	private static ArrayList<ArrayList<Integer>> getBodyNegationIdxs(String bodyId,
			FileHashMap<Integer, List<String>> trainBMap) throws IOException {

		int bIdx = Integer.valueOf(bodyId);
		List<String> sentDeps = trainBMap.get(bIdx);

		ArrayList<ArrayList<Integer>> negIdxs = new ArrayList<>();

		// System.out.println("sentDeps.size = " + sentDeps.size());
		for (String ss : sentDeps) {
			String[] deps = ss.split("\n");

			ArrayList<Integer> sentNegIdxs = new ArrayList<>();
			for (String d : deps) {
				String depType = d.substring(0, d.indexOf('('));

				if (depType.equals("neg")) {
					String betweenBrack = d.substring(d.indexOf('(') + 1, d.indexOf(')'));
					String[] depWords = betweenBrack.split(", ");

					//String negWord = depWords[0].substring(0, depWords[0].lastIndexOf('-'));

					int negWordIndex = Integer.valueOf(depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim());

					sentNegIdxs.add(negWordIndex);

				}
			}
			negIdxs.add(sentNegIdxs);
		}

		// System.out.println("negIdxs.size = " + negIdxs.size() + negIdxs);
		return negIdxs;
	}

	public static void saveNegFeaturesAsHashFile(String csvfilePath, String hashFileName) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, Integer> negData = new FileHashMap<String, Integer>(hashFileName, FileHashMap.FORCE_OVERWRITE);
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

	public static FileHashMap<String, Integer> loadNegFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Integer> negData = new FileHashMap<String, Integer>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return negData;
	}

}
