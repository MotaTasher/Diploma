import Code.Dataloader as DataloaderLib
import Code.Logger as LogsLib

import importlib
importlib.reload(DataloaderLib)
importlib.reload(LogsLib)


last_ind = int(2e7 + 1e6)
size = int(1e7)
DataloaderLib.GetTransactionsByInd(range(last_ind, last_ind - size, -1), logger_fake='logger_loader')
