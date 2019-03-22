
def create_model(opt,pretrained=None):
    model = None
    from .HG_model import HGModel
    model = HGModel(opt,pretrained)
    # print("model [%s] was created" % (model.name()))
    return model
