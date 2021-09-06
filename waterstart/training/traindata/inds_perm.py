import numpy as np
import numpy.typing as npt

from typing import no_type_check


@no_type_check
def compute_perm(
    n_batches: int, batch_size: int, min_gap: int = 1
) -> npt.NDArray[np.int64]:
    perm = np.random.permutation(n_batches * batch_size)
    all_inds = np.arange(perm.size)

    perm_view = perm.view()
    perm_view.shape = (n_batches, batch_size)
    last_n_invalids = np.inf

    while True:
        perm_view.sort(1)
        invalid_mask = perm_view[:, 1:] - perm_view[:, :-1] < min_gap

        invalid_inds = invalid_mask.nonzero()
        invalid_inds = np.ravel_multi_index(invalid_inds, perm_view.shape)
        n_invalids = invalid_inds.size

        if n_invalids == 0:
            break

        shuffle_inds = invalid_inds

        if n_invalids == 1 or n_invalids >= last_n_invalids:
            # print("bla!")
            valid_inds = np.setdiff1d(all_inds, invalid_inds, assume_unique=True)
            new_ind = np.random.choice(valid_inds, size=1)
            shuffle_inds = np.concatenate((shuffle_inds, new_ind))

        shuffle_perm = np.random.permutation(shuffle_inds.size)
        perm[shuffle_inds] = perm[shuffle_inds][shuffle_perm]

        last_n_invalids = n_invalids
        # print(n_invalids)

    return perm_view
