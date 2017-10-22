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

import ude.master.thesis.stance_detection.util.PPDBProcessorMidArguments;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class NegationFeaturesGeneratorMidArguments {

	private static Lemmatizer lemm;
	private static Porter porter;
	private static BufferedWriter out;
	private static FileHashMap<Integer, Map<Integer, List<String>>> trainBMap;
	private static FileHashMap<String, List<String>> trainTMap;
	private static FileHashMap<Integer, Map<Integer, List<String>>> testBMap;
	private static FileHashMap<String, List<String>> testTMap;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {

		//lemm = new Lemmatizer();
		//porter = new Porter();

		getNegFeaturesForData();

		 saveNegFeaturesAsHashFile(ProjectPaths.CSV_NEG_PARTS_ARG_FEATURE_TRAIN,
		 ProjectPaths.NEG_PARTS_ARG_FEATURE_TRAIN);
		 saveNegFeaturesAsHashFile(ProjectPaths.CSV_NEG_PARTS_ARG_FEATURE_TEST,
		 ProjectPaths.NEG_PARTS_ARG_FEATURE_TEST);

	}

	private static void getNegFeaturesForData() throws IOException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES_TRAIN,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.ARGUMENTED_MID_BODIES_TEST);

		trainBMap = getBodiesDepMap(ProjectPaths.TRAIN_BODIES_PARTS_ARG_DEPS);
		trainTMap = getTitlesDepMap(ProjectPaths.TRAIN_TITLES_DEPS2);// no need
																		// to
																		// get
																		// new
																		// ones

		testBMap = getBodiesDepMap(ProjectPaths.TEST_BODIES_PARTS_ARG_DEPS);
		testTMap = getTitlesDepMap(ProjectPaths.TEST_TITLES_DEPS2);
		System.out.println("trainTMap.size" + trainTMap.size());
		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES_TRAIN));
		List<List<String>> trainingStances = sddr.getTrainStances();

		generateNegFeaturesAndSave(trainingSummIdBoyMap, trainingStances, trainTMap, trainBMap,
				ProjectPaths.CSV_NEG_PARTS_ARG_FEATURE_TRAIN, ProjectPaths.PPDB_HUNG_SCORES_IDXS_PARTS_ARG_TRAIN);

		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.ARGUMENTED_MID_BODIES_TEST));
		List<List<String>> testStances = sddr.getTestStances();
		generateNegFeaturesAndSave(testSummIdBoyMap, testStances, testTMap, testBMap,
				ProjectPaths.CSV_NEG_PARTS_ARG_FEATURE_TEST, ProjectPaths.PPDB_HUNG_SCORES_IDXS_PARTS_ARG_TEST);

	}

	private static FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>> loadHungarianScores(
			String hungarianScorePath) {
		FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>> hung_scores = PPDBProcessorMidArguments
				.loadHungarianScoreIdxsFromFileMap(hungarianScorePath);
		return hung_scores;
	}

	private static FileHashMap<Integer, Map<Integer, List<String>>> getBodiesDepMap(String path) {
		FileHashMap<Integer, Map<Integer, List<String>>> bodiesDeps = null;
		try {
			bodiesDeps = new FileHashMap<Integer, Map<Integer, List<String>>>(path, FileHashMap.RECLAIM_FILE_GAPS);
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
	 * @param summIdBoyMap
	 * @param stances
	 * @param featuresFilename
	 *            file name to save the features to
	 * @throws IOException
	 */
	private static void generateNegFeaturesAndSave(HashMap<Integer, Map<Integer, String>> summIdBoyMap,
			List<List<String>> stances, FileHashMap<String, List<String>> titleDepMap,
			FileHashMap<Integer, Map<Integer, List<String>>> bodyDepMap, String featuresFilename,
			String hungarianScorePath) throws IOException {
		List<String[]> entries = new ArrayList<>();
		entries.add(new String[] { "title", "Body ID", "Stance", "neg_feature" });

		FileHashMap<String, Map<Integer, ArrayList<ArrayList<Integer>>>> hungScores = loadHungarianScores(
				hungarianScorePath);
		int i = 0;
		for (List<String> stance : stances) {

			List<String> entry = new ArrayList<>();
			entry.add(stance.get(0));
			entry.add(stance.get(1));
			entry.add(stance.get(2));
			
			//exclude bodies with less than 10 sentences (which are already removed from summIdBoyMap)
			if (summIdBoyMap.containsKey(Integer.valueOf(stance.get(1)))) {
				// System.out.println(stance.get(1));
				List<Integer> titleNegIdxs = getTitleNegationIdxs(stance.get(0), titleDepMap);

				// the bodyNegIdxs size should be equal to the sentences count
				// in the article
				ArrayList<ArrayList<Integer>> bodyNegIdxs = getBodyNegationIdxs(stance.get(1), bodyDepMap);

				for (int j = 0; j < bodyNegIdxs.size(); j++) {
					int negFeature = 0;
					if ((titleNegIdxs.size() == 0) && (bodyNegIdxs.get(j).size() == 0)) {
						entry.add(String.valueOf(negFeature));

						continue;
					}

					Map<Integer, ArrayList<ArrayList<Integer>>> idxsMap = hungScores.get(stance.get(0) + stance.get(1));

					// getidxs just for the first 5 and last 3 sentences
					// and here get the idxs just for one sentence
					ArrayList<Integer> hScoreidxs = new ArrayList<>();
					if ((j >= 0) && (j < 5)) {
						hScoreidxs = idxsMap.get(1).get(j);
					} else if ((j >= 6) && (j < 9)) {
						hScoreidxs = idxsMap.get(3).get(j - 6);
					} else if (j >= 10) {
						hScoreidxs = idxsMap.get(2).get(j - 10);
					}

					// System.out.println("idxs = " + idxs);
					// System.out.println("titleNegIdxs = " + titleNegIdxs);
					// System.out.println("bodyNegIdxs" + bodyNegIdxs);

					int tIdx = 0;

					for (tIdx = 0; tIdx < hScoreidxs.size(); tIdx++)
						if ((titleNegIdxs.contains(tIdx + 1) && !bodyNegIdxs.get(j).contains(hScoreidxs.get(tIdx) + 1))
								|| (!titleNegIdxs.contains(tIdx + 1)
										&& bodyNegIdxs.get(j).contains(hScoreidxs.get(tIdx) + 1)))
							negFeature++;

					entry.add(String.valueOf(negFeature));

				}

				entries.add(entry.toArray(new String[0]));
			}
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

	private static ArrayList<ArrayList<Integer>> getBodyNegationIdxs(String bodyId,
			FileHashMap<Integer, Map<Integer, List<String>>> bodyDepMap) throws IOException {

		int bIdx = Integer.valueOf(bodyId);
		Map<Integer, List<String>> partsDepMap = bodyDepMap.get(bIdx);
		// System.out.println(bodyId + " partsDepMap= " + partsDepMap);
		// System.out.println(bodyId + " partsDepMap.size()= " +
		// partsDepMap.size());
		ArrayList<ArrayList<Integer>> negIdxs = new ArrayList<>();

		for (int k = 1; k <= 3; k++) {
			if (k != 2) {
				List<String> sentDeps = partsDepMap.get(k);
				// System.out.println("sentDeps.size = " + sentDeps.size());
				for (String ss : sentDeps) {
					String[] deps = ss.split("\n");

					ArrayList<Integer> sentNegIdxs = new ArrayList<>();
					for (String d : deps) {
						String depType = d.substring(0, d.indexOf('('));

						if (depType.equals("neg")) {
							String betweenBrack = d.substring(d.indexOf('(') + 1, d.indexOf(')'));
							String[] depWords = betweenBrack.split(", ");

							String negWord = depWords[0].substring(0, depWords[0].lastIndexOf('-'));

							int negWordIndex = Integer
									.valueOf(depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim());

							sentNegIdxs.add(negWordIndex);

						}
					}
					negIdxs.add(sentNegIdxs);
				}

				/*
				 * if (k == 1) while (negIdxs.size() <
				 * BodySummarizerWithArguments.NUM_SENT_BEG) { negIdxs.add(new
				 * ArrayList<>()); } else if (k == 3) while (negIdxs.size() <
				 * BodySummarizerWithArguments.NUM_SENT_END +
				 * BodySummarizerWithArguments.NUM_SENT_BEG) { negIdxs.add(new
				 * ArrayList<>()); }
				 */
			}

		}

		//System.out.println("negIdxs.size = " + negIdxs.size() + negIdxs);

		for (int k = 1; k <= 3; k++) {
			if (k == 2) {
				List<String> sentDeps = partsDepMap.get(k);
				// System.out.println("sentDeps.size = " + sentDeps.size());
				for (String ss : sentDeps) {
					String[] deps = ss.split("\n");

					ArrayList<Integer> sentNegIdxs = new ArrayList<>();
					for (String d : deps) {
						String depType = d.substring(0, d.indexOf('('));

						if (depType.equals("neg")) {
							String betweenBrack = d.substring(d.indexOf('(') + 1, d.indexOf(')'));
							String[] depWords = betweenBrack.split(", ");

							String negWord = depWords[0].substring(0, depWords[0].lastIndexOf('-'));

							int negWordIndex = Integer
									.valueOf(depWords[0].substring(depWords[0].lastIndexOf('-') + 1).trim());

							sentNegIdxs.add(negWordIndex);

						}
					}
					negIdxs.add(sentNegIdxs);
				}
			}

		}
		//System.out.println("negIdxs.size = " + negIdxs.size() + negIdxs);
		return negIdxs;
	}

	public static void saveNegFeaturesAsHashFile(String csvfilePath, String hashFileName) throws FileNotFoundException,
			ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, int[]> negData = new FileHashMap<String, int[]>(hashFileName, FileHashMap.FORCE_OVERWRITE);
		CSVReader reader = null;
		reader = new CSVReader(new FileReader(csvfilePath));
		String[] line;
		line = reader.readNext();

		while ((line = reader.readNext()) != null) {
			int[] features = new int[line.length - 3];
			for (int i = 3; i < line.length; i++) {
				features[i-3] = Integer.valueOf(line[i]);
			}

			negData.put(line[0] + line[1], features);
		}
		reader.close();

		// saving the map file
		negData.save();
		negData.close();
	}

	public static FileHashMap<String, int[]> loadNegFeaturesAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, int[]> negData = new FileHashMap<String, int[]>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return negData;
	}

}
