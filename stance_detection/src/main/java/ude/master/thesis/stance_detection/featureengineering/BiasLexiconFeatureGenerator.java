package ude.master.thesis.stance_detection.featureengineering;

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
import java.util.Scanner;
import java.util.Set;
import java.util.Map.Entry;

import org.clapper.util.misc.FileHashMap;
import org.clapper.util.misc.ObjectExistsException;
import org.clapper.util.misc.VersionMismatchException;

import com.opencsv.CSVWriter;

import ude.master.thesis.stance_detection.processor.FeatureExtractorWithModifiedBL;
import ude.master.thesis.stance_detection.util.ProjectPaths;
import ude.master.thesis.stance_detection.util.StanceDetectionDataReader;

public class BiasLexiconFeatureGenerator {

	private static Set<String> biasSet;

	public static void main(String[] args)
			throws IOException, ObjectExistsException, ClassNotFoundException, VersionMismatchException {
		StanceDetectionDataReader sddr = new StanceDetectionDataReader(true, true,
				ProjectPaths.TRAIN_STANCES_PREPROCESSED, ProjectPaths.SUMMARIZED_TRAIN_BODIES2_WITH_MID,
				ProjectPaths.TEST_STANCESS_PREPROCESSED, ProjectPaths.SUMMARIZED_TEST_BODIES2_WITH_MID);

		List<List<String>> trainingStances = sddr.getTrainStances();
		HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TRAIN_BODIES2_WITH_MID));
		List<List<String>> testStances = sddr.getTestStances();
		HashMap<Integer, Map<Integer, String>> testSummIdBoyMap = sddr
				.readSummIdBodiesMap(new File(ProjectPaths.SUMMARIZED_TEST_BODIES2_WITH_MID));

		getBodiesBiasLexiconCount(trainingSummIdBoyMap, testSummIdBoyMap, ProjectPaths.BODY_BAIS_COUNT_TRAIN_TEST,
				ProjectPaths.CSV_BODY_BAIS_COUNT_TRAIN_TEST);

		getTitlesBiasLexiconCount(trainingStances, testStances, ProjectPaths.TITLE_BAIS_COUNT_TRAIN_TEST,
				ProjectPaths.CSV_TITLE_BAIS_COUNT_TRAIN_TEST);
	}

	/**
	 * 
	 * @param trainingStances
	 * @param testStances
	 * @param hashFileName
	 * @param csvFileName
	 * @throws FileNotFoundException
	 * @throws ObjectExistsException
	 * @throws ClassNotFoundException
	 * @throws VersionMismatchException
	 * @throws IOException
	 */
	private static void getTitlesBiasLexiconCount(List<List<String>> trainingStances, List<List<String>> testStances,
			String hashFileName, String csvFilePath) throws FileNotFoundException, ObjectExistsException,
			ClassNotFoundException, VersionMismatchException, IOException {

		FileHashMap<String, Integer> bias = new FileHashMap<String, Integer>(hashFileName, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[15];
		header[0] = "title";
		header[1] = "bias_num";

		entries.add(header);

		Set<String> allTitles = new HashSet<String>();
		for (List<String> s : trainingStances) {
			allTitles.add(s.get(0));
		}

		for (List<String> s : testStances) {
			allTitles.add(s.get(0));
		}

		for (String t : allTitles) {
			int biasCount = countBiasLexicons(t);
			bias.put(t, biasCount);

			List<String> entry = new ArrayList<>();
			entry.add(t);
			entry.add(String.valueOf(biasCount));
			entries.add(entry.toArray(new String[0]));
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

		bias.save();
		bias.close();

	}

	private static int countBiasLexicons(String t) throws FileNotFoundException {
		if (biasSet == null)
			initBiasLexicons(ProjectPaths.BIAS_LEX_PATH);

		String lemmas = FeatureExtractorWithModifiedBL.getLemmatizedCleanStr(t);

		int count = 0;
		for (String l : lemmas.split(" "))
			if (biasSet.contains(l))
				count++;
		return count;
	}

	private static void getBodiesBiasLexiconCount(HashMap<Integer, Map<Integer, String>> trainingSummIdBoyMap,
			HashMap<Integer, Map<Integer, String>> testSummIdBoyMap, String hashFilePath, String csvFilePath)
			throws ObjectExistsException, ClassNotFoundException, VersionMismatchException, IOException {
		FileHashMap<String, Integer> bias = new FileHashMap<String, Integer>(hashFilePath, FileHashMap.FORCE_OVERWRITE);

		List<String[]> entries = new ArrayList<>();

		String[] header = new String[15];
		header[0] = "body_id";
		header[1] = "bias_num";

		entries.add(header);

		Map<Integer, Map<Integer, String>> allBodyId = new HashMap<Integer, Map<Integer, String>>();
		allBodyId.putAll(trainingSummIdBoyMap);
		allBodyId.putAll(testSummIdBoyMap);

		for (Entry<Integer, Map<Integer, String>> b : allBodyId.entrySet()) {

			int biasCount = 0;
			for (Entry<Integer, String> parts : b.getValue().entrySet()) {
				biasCount += countBiasLexicons(parts.getValue());
			}
			bias.put(String.valueOf(b.getKey()), biasCount);

			List<String> entry = new ArrayList<>();
			entry.add(String.valueOf(b.getKey()));
			entry.add(String.valueOf(biasCount));
			entries.add(entry.toArray(new String[0]));
		}

		try (CSVWriter writer = new CSVWriter(new FileWriter(csvFilePath))) {
			writer.writeAll(entries);
		}

		bias.save();
		bias.close();

	}

	public static void initBiasLexicons(String filePath) throws FileNotFoundException {
		biasSet = new HashSet<>();
		Scanner s = new Scanner(new FileReader(filePath));
		while (s.hasNext())
			biasSet.add(s.next());
		s.close();
	}

	public static FileHashMap<String, Integer> loadBiasLexCountAsHashFile(String hashFileName)
			throws FileNotFoundException, ObjectExistsException, ClassNotFoundException, VersionMismatchException,
			IOException {
		FileHashMap<String, Integer> bSent = new FileHashMap<String, Integer>(hashFileName,
				FileHashMap.RECLAIM_FILE_GAPS);
		return bSent;
	}
}
