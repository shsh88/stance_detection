package ude.master.thesis.stance_detection.util;

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.hamcrest.collection.IsMapContaining;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import com.opencsv.CSVWriter;

import static org.hamcrest.MatcherAssert.assertThat;

public class StanceDetectionDataReaderTest {

	protected final String[] trainStancesSample = { "Headline", "Body ID", "Stance",

			"Police find mass graves with at least '15 bodies' near Mexico town where 43 students disappeared after police clash",
			"712", "unrelated",

			"Hundreds of Palestinians flee floods in Gaza as Israel opens dams", "158", "agree",

			"Spider burrowed through tourist's stomach and up into his chest", "1923", "disagree",

			"British Aid Worker Confirmed Murdered By ISIS,882,unrelated Gateway Pundit", "2327", "discuss" };

	protected final String[] trainBodiesSample = { "Body ID", "articleBody", "712",
			"\"Danny Boyle is directing the untitled "
					+ "film Seth Rogen is being eyed to play Apple co-founder Steve Wozniak in Sony’s "
					+ "Steve Jobs biopic. Danny Boyle is directing the untitled film, based on Walter "
					+ "Isaacson's book and adapted by Aaron Sorkin, which is one of the most anticipated "
					+ "biopics in recent years. Negotiations have not yet begun, and it’s not even "
					+ "clear if Rogen has an official offer, but the producers — Scott Rudin, Guymon "
					+ "Casady and Mark Gordon — have set their sights on the talent and are in talks. ",

			"158",
			"\"Hundreds of Palestinians were evacuated from their homes Sunday morning after "
					+ "Israeli authorities opened a number of dams near the border, flooding the Gaza "
					+ "Valley in the wake of a recent severe winter storm. The Gaza Ministry of Interior "
					+ "said in a statement that civil defense services and teams from the Ministry of Public "
					+ "Works had evacuated more than 80 families from both sides of the Gaza Valley \"",

			"1923",
			"\"Fear not arachnophobes, the story of Bunbury's \"\"spiderman\"\" might not be "
					+ "all it seemed. Perth scientists have cast doubt over claims that a spider burrowed into "
					+ "a man's body during his first trip to Bali. The story went global on Thursday, "
					+ "generating hundreds of stories online. Earlier this month, Dylan Thomas headed to "
					+ "the holiday island and sought medical help after experiencing \"\"a really burning sensation "
					+ "like a searing feeling\"\" in his abdomen.\"",

			"882",
			"\"The British Islamic State militant who has featured in videos featuring the "
					+ "execution of Western hostages, known as ‘Jihadi John’, has been identified. The man "
					+ "is Mohammed Emwazi, a young British man from West London who was known to British "
					+ "security services.\"" };

	List<List<String>> expectedTrainStances;
	Map<Integer, String> expectedTrainIdBodies;

	@Rule
	public TemporaryFolder tempFolder = new TemporaryFolder();

	private File tempTrainStancesFile;
	private File tempTrainBodiesFile;

	@Before
	public void getTestData() throws IOException {
		tempTrainStancesFile = tempFolder.newFile("train_stances.csv");

		CSVWriter csvWriter = new CSVWriter(new FileWriter(tempTrainStancesFile), ',');

		expectedTrainStances = new ArrayList<>();

		for (int i = 0; i < trainStancesSample.length; i += 3) {
			String[] line = { trainStancesSample[i], trainStancesSample[i + 1], trainStancesSample[i + 2] };
			if (i > 0)
				expectedTrainStances.add(Arrays.asList(line));
			csvWriter.writeNext(line, false);
		}
		csvWriter.close();

		tempTrainBodiesFile = tempFolder.newFile("train_bodies.csv");
		csvWriter = new CSVWriter(new FileWriter(tempTrainBodiesFile), ',');

		expectedTrainIdBodies = new HashMap<>();
		for (int i = 0; i < trainBodiesSample.length; i += 2) {
			String[] line = { trainBodiesSample[i], trainBodiesSample[i + 1] };
			if (i > 0)
				expectedTrainIdBodies.put(Integer.valueOf(trainBodiesSample[i]), trainBodiesSample[i + 1]);
			csvWriter.writeNext(line, false);
		}
		csvWriter.close();
	}

	/*
	 * @BeforeClass public static void setUpBeforeClass() throws Exception { }
	 */

	@Test
	public void test() throws IOException {

		StanceDetectionDataReader sdr = new StanceDetectionDataReader(true, false, tempTrainStancesFile.getPath(),
				tempTrainBodiesFile.getPath(), "", "");

		//System.out.println(sdr.getTrainStances().toString());
		//System.out.println(expectedTrainStances);
		//sdr.getTestIdBodyMap();

		assertEquals(sdr.getTrainStances().size(), expectedTrainStances.size());
		int trainStancesSize = expectedTrainStances.size();
		for (int i = 0; i < trainStancesSize; i++) {
			assertArrayEquals(sdr.getTrainStances().get(i).toArray(), expectedTrainStances.get(i).toArray());
		}
		
		assertEquals(sdr.getTrainIdBodyMap().size(), expectedTrainIdBodies.size());
		
		for(Entry<Integer, String> articl: expectedTrainIdBodies.entrySet()){
			assertThat(sdr.getTrainIdBodyMap(), IsMapContaining.hasEntry(articl.getKey(), articl.getValue()));
		}

	}

}
