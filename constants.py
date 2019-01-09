
NORM_DOC_LENGTH = 250
MIN_DOC_LENGTH = 500

# Document categories
EARN = "earn"
ACQ = "acq"
CRUDE = "crude"
CORN = "corn"
TRAIN = "train"
TEST = "test"

CATEGORIES = [EARN, ACQ, CRUDE, CORN]
DATA_SPLIT = [TRAIN, TEST]

SPLIT_SIZES = {
    TRAIN:{
        EARN:154, 
        ACQ:114,
        CRUDE:76,
        CORN:38
    },
    TEST:{
        EARN:40, 
        ACQ:25,
        CRUDE:15,
        CORN:10
    }
}