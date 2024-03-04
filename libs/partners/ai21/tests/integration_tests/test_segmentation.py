from langchain_ai21 import AI21SegmentationTextSplitter

TEXT = """The original full name of the franchise is Pocket Monsters (ポケットモンスター, Poketto Monsutā), which was abbreviated to Pokemon during development of the original games.
When the franchise was released internationally, the short form of the title was used, with an acute accent (´) over the e to aid in pronunciation.
Pokémon refers to both the franchise itself and the creatures within its fictional universe.
As a noun, it is identical in both the singular and plural, as is every individual species name;[10] it is grammatically correct to say "one Pokémon" and "many Pokémon", as well as "one Pikachu" and "many Pikachu".
In English, Pokémon may be pronounced either /'powkɛmon/ (poe-keh-mon) or /'powkɪmon/ (poe-key-mon).
The Pokémon franchise is set in a world in which humans coexist with creatures known as Pokémon.
Pokémon Red and Blue contain 151 Pokémon species, with new ones being introduced in subsequent games; as of December 2023, 1,025 Pokémon species have been introduced.
[b] Most Pokémon are inspired by real-world animals;[12] for example, Pikachu are a yellow mouse-like species[13] with lightning bolt-shaped tails[14] that possess electrical abilities.[15]
The player character takes the role of a Pokémon Trainer.
The Trainer has three primary goals: travel and explore the Pokémon world; discover and catch each Pokémon species in order to complete their Pokédex; and train a team of up to six Pokémon at a time and have them engage in battles.
Most Pokémon can be caught with spherical devices known as Poké Balls.
Once the opposing Pokémon is sufficiently weakened, the Trainer throws the Poké Ball against the Pokémon, which is then transformed into a form of energy and transported into the device.
Once the catch is successful, the Pokémon is tamed and is under the Trainer's command from then on.
If the Poké Ball is thrown again, the Pokémon re-materializes into its original state.
The Trainer's Pokémon can engage in battles against opposing Pokémon, including those in the wild or owned by other Trainers.
Because the franchise is aimed at children, these battles are never presented as overtly violent and contain no blood or gore.[I]
Pokémon never die in battle, instead fainting upon being defeated.[20][21][22]
After a Pokémon wins a battle, it gains experience and becomes stronger.[23] After gaining a certain amount of experience points, its level increases, as well as one or more of its statistics.
As its level increases, the Pokémon can learn new offensive and defensive moves to use in battle.[24][25] Furthermore, many species can undergo a form of spontaneous metamorphosis called Pokémon evolution, and transform into stronger forms.[26] Most Pokémon will evolve at a certain level, while others evolve through different means, such as exposure to a certain item.[27]
"""


def test_invoke__split_text_to_document() -> None:
    segmentation = AI21SegmentationTextSplitter()
    segments = segmentation.split_text_to_documents(
        source=TEXT)
    assert len(segments) > 0
    for segment in segments:
        assert segment.page_content is not None
        assert segment.metadata is not None


def test_invoke__split_text() -> None:
    segmentation = AI21SegmentationTextSplitter()
    segments = segmentation.split_text(
        source=TEXT)
    assert len(segments) > 0
