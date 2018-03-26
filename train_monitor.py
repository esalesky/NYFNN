from abc import ABC, abstractmethod
from utils import time_elapsed, perplexity, save_plot, MODEL_PATH

import logging
import time

logger = logging.getLogger(__name__)

"""Abstract class for implementing training callbacks."""


class TrainCallback(ABC):

    def __init__(self, iters_per_epoch, loss_type):
        self.iters_per_epoch = iters_per_epoch
        self.loss_type = loss_type
        self.iters = 0
        pass

    @abstractmethod
    def start_training(self):
        pass

    @abstractmethod
    def finish_iter(self, loss_type, loss):
        pass

    @abstractmethod
    def finish_epoch(self, epoch, loss_type, avg_loss, total_loss):
        pass

    @abstractmethod
    def finish_training(self):
        pass


"""Callback for printing loss during training. Default will only print loss information provided at end of each epoch. 
By specifying print_every, it will also print out loss and perplexity information after every X iterations."""


class PrintCallback(TrainCallback):

    def __init__(self, iters_per_epoch, loss_type, print_every=0, print_perplexity=True):
        super().__init__(iters_per_epoch, loss_type)
        self.print_every = print_every
        self.interval_loss_total = 0
        self.start = 0
        self.print_perplexity = print_perplexity

    def start_training(self):
        self.start = time.time()

    def finish_iter(self, loss_type, loss):
        if self.loss_type != loss_type:
            return
        if self.print_every > 0:
            self.interval_loss_total += loss
            self.iters += 1
            if self.iters % self.print_every == 0:
                interval_loss_avg = self.interval_loss_total / self.print_every
                perc_through_epoch = self.iters / self.iters_per_epoch
                logger.info('Batch: {} / {}. {}'.format(self.iters, self.iters_per_epoch, time_elapsed(self.start,
                                                                                                   perc_through_epoch)))
                logger.info('\tLoss: {0:.4f}'.format(interval_loss_avg))
                if self.print_perplexity:
                    print_perplexity_avg = perplexity(interval_loss_avg)
                    logger.info('\tPerplexity: {0:.4f}'.format(print_perplexity_avg))
                self.interval_loss_total = 0

    def finish_epoch(self, epoch, loss_type, avg_loss, total_loss):
        self.iters = 0
        self.interval_loss_total = 0
        self.start = time.time()
        if loss_type != self.loss_type:
            return
        ppl = perplexity(avg_loss)
        logger.info("-" * 65)
        logger.info("Epoch {}: {} ppl, {:.4f}. avg loss, {:.4f}. total loss, {:.4f}".format(epoch, self.loss_type, ppl,
                                                                                            avg_loss, total_loss))
        logger.info("-" * 65)
        return "continue"


    def finish_training(self):
        pass

"""Callback for plotting loss and optionally perplexity. Note that separate callbacks should be created for
   plotting the same loss (i.e. test or dev) at different intervals, such as every epoch or every 10,000 iterations."""


class PlotCallback(TrainCallback):

    def __init__(self, iters_per_epoch, loss_type, loss_file, plot_every=0, plot_scale=1, save_every=0,
                 perplexity_file=None):
        super().__init__(iters_per_epoch, loss_type)
        self.loss_file = loss_file
        self.plot_every = plot_every
        self.save_every = save_every
        self.plot_losses = []
        self.plot_loss_total = 0
        # X-axis scale, allows for plotting values that are pre-averaged
        self.plot_scale = plot_scale * self.plot_every
        if perplexity_file:
            self.perplexity_file = perplexity_file
            self.plot_perplexities = []

    def start_training(self):
        pass

    def finish_iter(self, loss_type, loss):
        if self.loss_type != loss_type:
            return
        if self.plot_every > 0:
            self.plot_loss_total += loss
            self.iters += 1
            if self.iters % self.plot_every == 0:
                plot_loss_avg = self.plot_loss_total / self.plot_every
                self.plot_losses.append(plot_loss_avg)
                self.plot_loss_total = 0
                if self.perplexity_file:
                    plot_perplexity_avg = perplexity(plot_loss_avg)
                    self.plot_perplexities.append(plot_perplexity_avg)
            if self.iters % self.save_every == 0:
                save_plot(self.plot_losses, self.loss_file, self.plot_scale)
                if self.plot_perplexities:
                    save_plot(self.plot_perplexities, self.perplexity_file, self.plot_scale)

    def finish_epoch(self, epoch, loss_type, avg_loss, total_loss):
        # Note that we explicitly don't reset self.iters here, because it would mess up how often we plot
        if self.loss_type != loss_type:
            return
        self.plot_losses.append(avg_loss)
        save_plot(self.plot_losses, self.loss_file, 1)
        if self.perplexity_file:
            self.plot_perplexities.append(perplexity(avg_loss))
            save_plot(self.plot_perplexities, self.perplexity_file, 1)
        return "continue"

    def finish_training(self):
        # Shouldn't need to do anything at this point, since graphs have fixed intervals for plotting
        pass



"""Bootstrap the TrainCallback abstraction to save a model every X epochs when a particular loss
   is calculated. Defaults to saving the model every epoch, can configure to save less frequently with save_every."""

class SaveModelCallback(TrainCallback):

    def __init__(self, iters_per_epoch, loss_type, model, model_path='/models/', save_every=0, patience=5, num_epochs=30):
        super().__init__(iters_per_epoch, loss_type)
        self.model = model
        self.save_every = save_every  #not used now, save every 1 ep if loss goes down
        self.model_path = model_path
        self.total_loss = 0
        self.num_epochs = num_epochs
        self.patience=patience     #num epochs to wait for dev loss to go down (below global min)
        self.dev_loss=float('inf') #init to very large number
        self.epochs_elapsed=0      #num epochs since dev loss has gone down

    def start_training(self):
        pass

    def finish_iter(self, loss_type, loss):
        pass

    def finish_epoch(self, epoch, loss_type, avg_loss, total_loss):
        status = "continue"
        if loss_type != self.loss_type:
            return status
        
        if avg_loss < self.dev_loss:
            self.dev_loss = avg_loss
            self.epochs_elapsed = 0
            if epoch == self.num_epochs-1:  #final epoch
                t = time.strftime("%Y-%m-%d-%H%M%S", time.localtime(time.time()))
                model_name = "model_final_{}".format(t)
            else:
                model_name = "model_e{0}_loss{1:.3f}".format(epoch, avg_loss)
            self.model.save("{}{}.model".format(self.model_path, model_name))
        else:
            self.epochs_elapsed += 1
            logger.info("Not saving model ({}): dev loss went up".format(self.epochs_elapsed))

        if self.epochs_elapsed == self.patience:
            status = "done"
        return status

    def finish_training(self):
        pass


class TrainMonitor(TrainCallback):

    def __init__(self, model, iters_per_epoch, print_every=1000, plot_every=100, save_plot_every=100, model_every=10, checkpoint_every=1000, patience=5, num_epochs=30):
        self.checkpoint = checkpoint_every
        self.callbacks = []
        self.callbacks.append(PrintCallback(iters_per_epoch, 'train', print_every=print_every))
        # Print dev and test metrics once per epoch
        self.callbacks.append(PrintCallback(iters_per_epoch, 'dev'))
        self.callbacks.append(PrintCallback(iters_per_epoch, 'test'))
        # Plot train loss every plot_every epochs
        self.callbacks.append(PlotCallback(iters_per_epoch, 'train', 'train_loss', plot_every=plot_every,
                                           save_every=save_plot_every, perplexity_file='train_perplexity'))
        # Want to plot dev loss every time we compute it in trainer, so plot every time we run the checkpoint
        # self.callbacks.append(PlotCallback(iters_per_epoch, 'dev-cp', 'dev_loss', plot_every=1, plot_scale=checkpoint_every,
        #                                    save_every=1, perplexity_file='dev_perplexity'))
        # Plot dev loss and perplexity once per epoch
        self.callbacks.append(PlotCallback(iters_per_epoch, 'dev', 'dev_epoch_loss',
                                           perplexity_file='dev_epoch_perplexity'))
        # # Save model
        # self.callbacks.append(SaveModelCallback(iters_per_epoch, 'train', model, model_path=MODEL_PATH,
        #                                         save_every=checkpoint_every))
        # # Save model
        self.callbacks.append(SaveModelCallback(iters_per_epoch, 'dev', model, model_path=MODEL_PATH,
                                                save_every=model_every, patience=patience, num_epochs=num_epochs))


    def set_iters(self, iters_per_epoch):
        self.iters_per_epoch = iters_per_epoch
        for c in self.callbacks:
            c.iters_per_epoch = iters_per_epoch


    def start_training(self):
        for c in self.callbacks:
            c.start_training()

    def finish_iter(self, loss_type, loss):
        for c in self.callbacks:
            c.finish_iter(loss_type, loss)

    def finish_epoch(self, epoch, loss_type, avg_loss, total_loss):
        done = False
        for c in self.callbacks:
            status = c.finish_epoch(epoch, loss_type, avg_loss, total_loss)
            if status=="done":
                done = True
        return done
    
    def finish_training(self):
        for c in self.callbacks:
            c.finish_training()
        logger.info("Finished training.")
