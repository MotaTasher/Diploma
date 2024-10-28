import Code.Dataloader as dataloader
import Code.Logger as logs

import importlib
importlib.reload(dataloader)
importlib.reload(logs)


last_ind = int(2e7 + 1e6)
size = int(1e5)
dataloader.GetTransactionsByInd(range(last_ind, last_ind - size, -1), logger_fake='logger_loader')
