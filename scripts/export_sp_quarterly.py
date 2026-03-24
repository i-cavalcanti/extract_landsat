import ee
import yaml
from datetime import datetime

LANDSAT = {
    "LT05": "LANDSAT/LT05/C02/T1_L2",
    "LE07": "LANDSAT/LE07/C02/T1_L2",
    "LC08": "LANDSAT/LC08/C02/T1_L2",
    "LC09": "LANDSAT/LC09/C02/T1_L2",
}

def init_ee():
    try:
        ee.Initialize(project='satexport')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='satexport')

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_sp_geometry(cfg):
    # GAUL level1 (limite administrativo) — bom pra teste/escala SP
    gaul1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    admin0 = cfg["aoi_admin0"]
    variants = cfg["aoi_admin1_variants"]

    base = gaul1.filter(ee.Filter.eq("ADM0_NAME", admin0))
    filt = ee.Filter.eq("ADM1_NAME", variants[0])
    for v in variants[1:]:
        filt = ee.Filter.Or(filt, ee.Filter.eq("ADM1_NAME", v))

    feat = base.filter(filt).first()
    return ee.Feature(feat).geometry()

def apply_scale_factors(img):
    # SR: scale/offset
    optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    # ST: scale/offset (Kelvin) — pode não existir em imagens L2SR
    thermal = img.select("ST_B.*").multiply(0.00341802).add(149.0)
    return img.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

def mask_clouds_and_saturation(img):
    qa = img.select("QA_PIXEL")

    # Bits (QA_PIXEL):
    # 1: Dilated Cloud, 2: Cirrus, 3: Cloud, 4: Cloud Shadow, 5: Snow, 6: Clear
    clear = qa.bitwiseAnd(1 << 6).neq(0)
    not_dilated = qa.bitwiseAnd(1 << 1).eq(0)
    not_cirrus = qa.bitwiseAnd(1 << 2).eq(0)
    not_cloud = qa.bitwiseAnd(1 << 3).eq(0)
    not_shadow = qa.bitwiseAnd(1 << 4).eq(0)
    not_snow = qa.bitwiseAnd(1 << 5).eq(0)

    mask = clear.And(not_dilated).And(not_cirrus).And(not_cloud).And(not_shadow).And(not_snow)

    # Saturação
    sat = img.select("QA_RADSAT").eq(0)
    mask = mask.And(sat)

    return img.updateMask(mask)

def masked_constant(name):
    # banda totalmente mascarada (pra padronizar coleções quando ST não existe)
    return ee.Image.constant(0).rename(name).updateMask(ee.Image.constant(0))

def get_temp_band(img, sensor):
    # Retorna banda ST_* se existir; senão banda mascarada
    if sensor in ("LT05", "LE07"):
        band = "ST_B6"
    else:
        band = "ST_B10"

    has = img.bandNames().contains(band)
    tempK = ee.Image(ee.Algorithms.If(has, img.select(band), masked_constant(band)))
    return tempK.rename("tempK")

def maybe_filter_stqa(img, use_filter, stqa_max):
    if not use_filter:
        return img
    # ST_QA pode não existir; se não existir, não filtra (mantém como está)
    has = img.bandNames().contains("ST_QA")
    stqa = ee.Image(ee.Algorithms.If(has, img.select("ST_QA"), masked_constant("ST_QA")))
    # ST_QA escala 0.01K -> comparar em unidade "centiKelvin"
    return img.updateMask(stqa.lte(stqa_max))

def add_ndvi_temp(img, sensor, cfg):
    img = apply_scale_factors(img)
    img = mask_clouds_and_saturation(img)

    if sensor in ("LT05", "LE07"):
        red = img.select("SR_B3")
        nir = img.select("SR_B4")
    else:
        red = img.select("SR_B4")
        nir = img.select("SR_B5")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")

    tempK = get_temp_band(img, sensor)
    tempC = tempK.subtract(273.15).rename("tempC")

    out = ee.Image.cat([ndvi, tempC])
    out = maybe_filter_stqa(out, cfg.get("use_st_qa_filter", False), cfg.get("st_qa_max", 200))

    return out.copyProperties(img, img.propertyNames())

def base_collection(collection_id, aoi, start, end, cloud_cover_land_max):
    return (ee.ImageCollection(collection_id)
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUD_COVER_LAND", cloud_cover_land_max)))

def landsat_quarter_collection(aoi, start, end, cfg):
    maxcc = cfg["cloud_cover_land_max"]

    lt05 = base_collection(LANDSAT["LT05"], aoi, start, end, maxcc).map(lambda i: add_ndvi_temp(i, "LT05", cfg))
    le07 = base_collection(LANDSAT["LE07"], aoi, start, end, maxcc).map(lambda i: add_ndvi_temp(i, "LE07", cfg))
    lc08 = base_collection(LANDSAT["LC08"], aoi, start, end, maxcc).map(lambda i: add_ndvi_temp(i, "LC08", cfg))
    lc09 = base_collection(LANDSAT["LC09"], aoi, start, end, maxcc).map(lambda i: add_ndvi_temp(i, "LC09", cfg))

    return lt05.merge(le07).merge(lc08).merge(lc09)

def quarterly_composite(year, quarter, aoi, cfg):
    start_month = (quarter - 1) * 3 + 1
    start = ee.Date.fromYMD(year, start_month, 1)
    end = start.advance(3, "month")

    col = landsat_quarter_collection(aoi, start, end, cfg)

    ndvi_med = col.select("ndvi").median().rename("ndvi")
    temp_med = col.select("tempC").median().rename("tempC")

    n_ndvi = col.select("ndvi").count().rename("n_ndvi")
    n_temp = col.select("tempC").count().rename("n_temp")

    # flag bitmask:
    # 1 => NDVI com pouca amostragem
    # 2 => TEMP com pouca amostragem
    ndvi_bad = n_ndvi.lt(cfg["n_obs_min_ndvi"]).multiply(1)
    temp_bad = n_temp.lt(cfg["n_obs_min_temp"]).multiply(2)
    flag = ndvi_bad.add(temp_bad).rename("flag")

    out = ee.Image.cat([ndvi_med, temp_med, n_ndvi, n_temp, flag]).clip(aoi)

    return out.set({
        "year": year,
        "quarter": quarter,
        "start_date": start.format("YYYY-MM-dd"),
        "end_date": end.advance(-1, "day").format("YYYY-MM-dd"),
        "cloud_cover_land_max": cfg["cloud_cover_land_max"],
        "n_obs_min_ndvi": cfg["n_obs_min_ndvi"],
        "n_obs_min_temp": cfg["n_obs_min_temp"],
        "system:time_start": start.millis(),
    })

def export_image(img, region, cfg, year, quarter):
    prefix = cfg["file_prefix"]
    name = f"{prefix}_{year}_Q{quarter}"
    scale = cfg["scale"]
    max_pixels = int(cfg.get("max_pixels", 10_000_000_000_000))

    target = cfg.get("export_target", "drive").lower()
    if target == "drive":
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=name,
            folder=cfg["drive_folder"],
            fileNamePrefix=name,
            region=region,
            scale=scale,
            maxPixels=max_pixels
        )
    elif target == "gcs":
        # precisa bucket + (opcional) path
        bucket = cfg["gcs_bucket"]
        gcs_path = cfg.get("gcs_path", "")
        task = ee.batch.Export.image.toCloudStorage(
            image=img,
            description=name,
            bucket=bucket,
            fileNamePrefix=f"{gcs_path}{name}",
            region=region,
            scale=scale,
            maxPixels=max_pixels
        )
    else:
        raise ValueError("export_target deve ser 'drive' ou 'gcs'.")

    task.start()
    return task

def main():
    init_ee()
    cfg = load_config("config.yaml")

    aoi = get_sp_geometry(cfg)

    start_year = cfg["start_year"]
    end_year = cfg["end_year"]
    quarters = cfg["quarters"]

    tasks = []
    for year in range(start_year, end_year + 1):
        for q in quarters:
            img = quarterly_composite(year, q, aoi, cfg)
            task = export_image(img, aoi, cfg, year, q)
            tasks.append(task)
            print(f"Started: {cfg['file_prefix']}_{year}_Q{q} | task_id={task.id}")

    print(f"\nTotal tasks started: {len(tasks)}")
    print("Dica: monitore em https://code.earthengine.google.com/ (aba Tasks) ou via ee.batch.Task.list()")

if __name__ == "__main__":
    main()