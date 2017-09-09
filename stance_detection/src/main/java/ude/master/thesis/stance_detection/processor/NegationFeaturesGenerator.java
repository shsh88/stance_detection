package ude.master.thesis.stance_detection.processor;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.lucene.util.SameThreadExecutorService;
import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.simple.Sentence;
import ude.master.thesis.stance_detection.util.PPDBProcessor;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class NegationFeaturesGenerator {

	private static Lemmatizer lemm;
	private static Porter porter;
	private static BufferedWriter out;
	private static FileHashMap<Integer, List<String>> trainBMap;
	private static FileHashMap<String, List<String>> trainTMap;
	private static FileHashMap<Integer, List<String>> testBMap;
	private static FileHashMap<String, List<String>> testTMap;

	public static void main(String[] args) throws IOException {
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
		out = new BufferedWriter(new FileWriter("resources/words_not_processed.txt"));
		out.write("==========================");
		out.newLine();

		lemm = new Lemmatizer();
		porter = new Porter();
		getNegFeaturesForData();
		out.flush();
		out.close();
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
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true, "resources/data/train_stances.csv",
				"resources/data/summ_train_bodies.csv", "resources/data/test_data/competition_test_stances.csv",
				"resources/data/test_data/summ_competition_test_bodies.csv");

		trainBMap = getBodiesDepMap("C:/thesis_stuff/help_files/train_bodies_deps");
		trainTMap = getTitlesDepMap("C:/thesis_stuff/help_files/train_titles_deps");

		testBMap = getBodiesDepMap("C:/thesis_stuff/help_files/test_bodies_deps");
		testTMap = getTitlesDepMap("C:/thesis_stuff/help_files/test_titles_deps");

		Map<Integer, String> trainIdBodyMap = sddr.getTrainIdBodyMap();
		List<List<String>> trainingStances = sddr.getTrainStances();
		// generateNegFeaturesAndSave(trainIdBodyMap, trainingStances,
		// "C:/thesis_stuff/features/train_neg_features.csv",
		// "train");
		Map<Integer, String> testIdBodyMap = sddr.getTestIdBodyMap();
		List<List<String>> testStances = sddr.getTestStances();
		generateNegFeaturesAndSave(testIdBodyMap, testStances, "C:/thesis_stuff/features/test_neg_features.csv", "test");

	}

	private static FileHashMap<String, ArrayList<Integer>> loadHungarianScores() {
		FileHashMap<String, ArrayList<Integer>> hung_scores = PPDBProcessor
				.loadHungarianScoreFromFileMap("C:/thesis_stuff/features/test_features/map_test_hung_ppdb");
		return hung_scores;
	}

	private static FileHashMap<Integer, List<String>> getBodiesDepMap(String path) {
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

	private static FileHashMap<String, List<String>> getTitlesDepMap(String path) {
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
	 * @param trainIdBodyMap
	 * @param trainingStances
	 * @param filename
	 *            file name to save the features to
	 * @throws IOException
	 */
	private static void generateNegFeaturesAndSave(Map<Integer, String> trainIdBodyMap,
			List<List<String>> trainingStances, String filename, String testTrainType) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "neg_feature" });

		FileHashMap<String, ArrayList<Integer>> hungScores = loadHungarianScores();
		System.out.println(trainingStances.size());
		int i = 0;
		for (List<String> stance : trainingStances) {

			List<String> entry = new ArrayList<>();
			entry.add(stance.get(0));
			entry.add(stance.get(1));
			entry.add(stance.get(2));

			// System.out.println(stance.get(1));
			List<Integer> titleNegIdxs = getNegationIdxs(stance.get(0), "title" + "-" + testTrainType);
			List<Integer> bodyNegIdxs = getNegationIdxs(
					stance.get(1) + "|" + trainIdBodyMap.get(Integer.valueOf(stance.get(1))),
					"body" + "-" + testTrainType);
			if ((titleNegIdxs.size() == 0) && (bodyNegIdxs.size() == 0)) {
				int negFeature = 0;

				entry.add(String.valueOf(negFeature));

				entries.add(entry.toArray(new String[0]));

				if (i % 10000 == 0)
					System.out.println("processed: " + i);
				i++;
				continue;
			}

			ArrayList<Integer> idxs = hungScores.get(stance.get(0) + stance.get(1));

			// System.out.println("idxs = " + idxs);
			// System.out.println("titleNegIdxs = " + titleNegIdxs);
			// System.out.println("bodyNegIdxs" + bodyNegIdxs);

			int tIdx = 0;
			int negFeature = 0;
			for (tIdx = 0; tIdx < idxs.size(); tIdx++)
				if ((titleNegIdxs.contains(tIdx) && !bodyNegIdxs.contains(idxs.get(tIdx)))
						|| (!titleNegIdxs.contains(tIdx) && bodyNegIdxs.contains(idxs.get(tIdx))))
					negFeature = 1;

			entry.add(String.valueOf(negFeature));

			entries.add(entry.toArray(new String[0]));

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

	/**
	 * 
	 * @param txt
	 *            when with bodies then txt is the id
	 * @param txtType
	 * @return
	 * @throws IOException
	 */
	private static List<Integer> getNegationIdxs(String txt, String txtType) throws IOException {

		List<String> graphs = null;
		if (txtType.equals("title-train"))
			graphs = trainTMap.get(txt);

		if (txtType.equals("body-train")) {
			int bIdx = Integer.valueOf(txt.substring(0, txt.indexOf('|')).trim());
			// System.out.println(txt.substring(0, txt.indexOf('|')).trim());
			graphs = trainBMap.get(bIdx);
			txt = txt.substring(txt.indexOf('|') + 1);
			// System.out.println(txt);
		}

		if (txtType.equals("title-test"))
			graphs = testTMap.get(txt);

		if (txtType.equals("body-test")) {
			int bIdx = Integer.valueOf(txt.substring(0, txt.indexOf('|')).trim());
			// System.out.println(txt.substring(0, txt.indexOf('|')).trim());
			graphs = testBMap.get(Integer.valueOf(bIdx));
			txt = txt.substring(txt.indexOf('|') + 1);
			// System.out.println(txt);
		}

		Map<String, Integer> lemmaMap = lemm.lemmatizeWithIdx(txt);
		// System.out.println(lemmaMap);

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
					// System.out.println("not word = " +
					// depWords[1].substring(0,
					// depWords[1].lastIndexOf('-')).trim());

					String negWord = lemm.lemmatize(depWords[0].substring(0, depWords[0].lastIndexOf('-')).trim())
							.get(0).toLowerCase();
					int negWordIndex = Integer.valueOf(depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim());

					// System.out.println(negWord + "," + i + "," +
					// negWordIndex);
					try {
						if (FeatureExtractor.isStopword(negWord)) {
							// System.out.println("neg word is stop word : " +
							// negWord);
							out.append("Stop word: " + negWord);
							out.newLine();
							continue;
						}
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					String key = porter.stripAffixes(negWord) + "," + i + "," + negWordIndex;
					if (lemmaMap.containsKey(key)) {
						int idx = lemmaMap.get(key);

						// System.out.println("negWord = " + negWord + " " +
						// idx);
						negIdxs.add(idx);
					} else {
						out.append(negWord + "  --->  " + txt);
						out.newLine();
						out.append(lemmaMap.toString());
						out.newLine();
					}
				}
			}
			i++;
		}

		return negIdxs;
	}

}