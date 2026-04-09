#!/usr/bin/env python3
"""
Pipeline anual para São Paulo com Landsat Collection 2 Level-2 no Google Earth Engine.

Inclui três etapas no mesmo arquivo:
1) export  : gera raster anual contínuo para o estado de SP, com saída em tiles no Drive
             ou como Asset no Earth Engine.
2) mosaic  : mosaica localmente os TIFFs exportados por ano.
3) clip    : recorta opcionalmente o mosaico anual por município.

Bandas anuais geradas:
- ndvi_max_annual
- ndvi_median_annual
- ndbi_median_annual
- tempC_mean_clear_annual
- tempC_hot_season_median_annual
- n_obs_annual
- mask_valid_annual

Exemplos de uso:

1) Exportar 2022 para Drive em tiles:
python gee_sp_annual_pipeline.py export \
  --project satexport \
  --shp /caminho/municipios.shp \
  --years 2022 \
  --export-folder gee_exports \
  --export-prefix landsat_annual_sp

2) Exportar 2000,2010,2022 para Asset:
python gee_sp_annual_pipeline.py export \
  --project satexport \
  --shp /caminho/municipios.shp \
  --years 2000 2010 2022 \
  --export-mode asset \
  --asset-root projects/seu-projeto-earthengine/assets/landsat_annual_sp

3) Mosaicar tiles baixados do Drive:
python gee_sp_annual_pipeline.py mosaic \
  --input-dir /caminho/tiffs \
  --output-dir /caminho/mosaicos

4) Recortar mosaicos por município:
python gee_sp_annual_pipeline.py clip \
  --mosaic-dir /caminho/mosaicos \
  --shp /caminho/municipios.shp \
  --output-dir /caminho/recortes \
  --cd-mun-field CD_MUN
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import ee
import geemap
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge


LANDSAT = {
    "LT05": "LANDSAT/LT05/C02/T1_L2",
    "LE07": "LANDSAT/LE07/C02/T1_L2",
    "LC08": "LANDSAT/LC08/C02/T1_L2",
    "LC09": "LANDSAT/LC09/C02/T1_L2",
}

NODATA_DEFAULT = -9999.0
HOT_MONTHS_1 = (1, 3)     # jan-mar
HOT_MONTHS_2 = (10, 12)   # out-dez
YEAR_PATTERN = re.compile(r"(\d{4})")


# ============================================================
# Helpers gerais
# ============================================================
def normalize_text(value) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .upper()
        .replace("Á", "A")
        .replace("À", "A")
        .replace("Ã", "A")
        .replace("Â", "A")
        .replace("É", "E")
        .replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ô", "O")
        .replace("Õ", "O")
        .replace("Ú", "U")
        .replace("Ç", "C")
    )


def parse_years(values: Iterable[str]) -> list[int]:
    years: list[int] = []
    for value in values:
        years.append(int(value))
    if not years:
        raise ValueError("Informe pelo menos um ano em --years.")
    return years


# ============================================================
# Earth Engine
# ============================================================
def init_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def sensors_for_year(year: int) -> list[str]:
    if year <= 2012:
        return ["LT05", "LE07"] if year >= 1999 else ["LT05"]
    if year <= 2021:
        return ["LC08"]
    return ["LC08", "LC09"]


def apply_scale_factors(img: ee.Image) -> ee.Image:
    optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = img.select("ST_B.*").multiply(0.00341802).add(149.0)
    return img.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)


def mask_clouds(img: ee.Image, sensor: str) -> ee.Image:
    qa = img.select("QA_PIXEL")

    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
        .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud shadow
        .And(qa.bitwiseAnd(1 << 5).eq(0))  # snow
    )

    if sensor in ("LC08", "LC09"):
        mask = mask.And(qa.bitwiseAnd(1 << 2).eq(0))  # cirrus

    sat = img.select("QA_RADSAT").eq(0)
    return img.updateMask(mask.And(sat))


def add_indices_temp(img: ee.Image, sensor: str) -> ee.Image:
    img = apply_scale_factors(img)
    img = mask_clouds(img, sensor)

    if sensor in ("LT05", "LE07"):
        ndvi = img.normalizedDifference(["SR_B4", "SR_B3"]).rename("ndvi")
        ndbi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndbi")
        temp_k = img.select("ST_B6")
    else:
        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
        ndbi = img.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi")
        temp_k = img.select("ST_B10")

    temp_c = temp_k.subtract(273.15).rename("tempC")
    return ee.Image.cat([ndvi, ndbi, temp_c]).copyProperties(img, img.propertyNames())


def base_collection(
    collection_id: str,
    region: ee.Geometry,
    start: ee.Date,
    end: ee.Date,
) -> ee.ImageCollection:
    return (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .filter(ee.Filter.lte("CLOUD_COVER_LAND", 80))
    )


# ============================================================
# Geometria de SP a partir do shapefile de municípios
# ============================================================
def load_sao_paulo_gdf(shp_path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("O shapefile foi lido, mas está vazio.")

    gdf = gdf[gdf.geometry.notnull()].copy()
    invalid_count = int((~gdf.is_valid).sum())
    print(f"Geometrias inválidas antes da correção: {invalid_count}")
    if invalid_count > 0:
        gdf["geometry"] = gdf.geometry.buffer(0)

    if "CD_UF" not in gdf.columns:
        raise ValueError(
            "O shapefile precisa ter a coluna CD_UF para selecionar São Paulo. "
            f"Campos disponíveis: {gdf.columns.tolist()}"
        )

    gdf["CD_UF_norm"] = gdf["CD_UF"].astype(str).str.strip()
    gdf_sp = gdf[gdf["CD_UF_norm"] == "35"].copy()

    if gdf_sp.empty:
        raise ValueError(
            "Nenhuma feição com CD_UF == '35' foi encontrada no shapefile."
        )

    print("Filtro do estado: CD_UF == '35'")
    return gdf_sp


def load_sao_paulo_ee_geometry(shp_path: str | Path) -> ee.Geometry:
    gdf_sp = load_sao_paulo_gdf(shp_path)
    print(f"Feições selecionadas para SP: {len(gdf_sp)}")
    state_gdf = gdf_sp.dissolve()
    return geemap.gdf_to_ee(state_gdf).geometry()


# ============================================================
# Composite anual
# ============================================================
def annual_collection(year: int, region: ee.Geometry) -> tuple[ee.ImageCollection, ee.Date, ee.Date]:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")

    merged = None
    for sensor in sensors_for_year(year):
        col = base_collection(LANDSAT[sensor], region, start, end).map(
            lambda i, s=sensor: add_indices_temp(i, s)
        )
        merged = col if merged is None else merged.merge(col)

    if merged is None:
        merged = ee.ImageCollection([])

    return merged, start, end


def hot_season_collection(col: ee.ImageCollection) -> ee.ImageCollection:
    part1 = col.filter(ee.Filter.calendarRange(HOT_MONTHS_1[0], HOT_MONTHS_1[1], "month"))
    part2 = col.filter(ee.Filter.calendarRange(HOT_MONTHS_2[0], HOT_MONTHS_2[1], "month"))
    return part1.merge(part2)


def empty_output_image(region: ee.Geometry, nodata: float) -> ee.Image:
    bands = [
        ee.Image.constant(nodata).rename("ndvi_max_annual").clip(region).toFloat(),
        ee.Image.constant(nodata).rename("ndvi_median_annual").clip(region).toFloat(),
        ee.Image.constant(nodata).rename("ndbi_median_annual").clip(region).toFloat(),
        ee.Image.constant(nodata).rename("tempC_mean_clear_annual").clip(region).toFloat(),
        ee.Image.constant(nodata).rename("tempC_hot_season_median_annual").clip(region).toFloat(),
        ee.Image.constant(0).rename("n_obs_annual").clip(region).toFloat(),
        ee.Image.constant(0).rename("mask_valid_annual").clip(region).toFloat(),
    ]
    return ee.Image.cat(bands).toFloat()


def annual_composite(
    year: int,
    search_geom: ee.Geometry,
    target_geom: ee.Geometry,
    nodata: float,
) -> tuple[ee.Image, int, int]:
    col, start, end = annual_collection(year, search_geom)
    annual_count = int(col.size().getInfo())

    if annual_count == 0:
        out = empty_output_image(target_geom, nodata)
        hot_count = 0
    else:
        ndvi_max = col.select("ndvi").max().rename("ndvi_max_annual").toFloat()
        ndvi_median = col.select("ndvi").median().rename("ndvi_median_annual").toFloat()
        ndbi_median = col.select("ndbi").median().rename("ndbi_median_annual").toFloat()
        temp_mean = col.select("tempC").mean().rename("tempC_mean_clear_annual").toFloat()

        hot_col = hot_season_collection(col)
        hot_count = int(hot_col.size().getInfo())

        if hot_count == 0:
            temp_hot_median = (
                ee.Image.constant(nodata)
                .rename("tempC_hot_season_median_annual")
                .clip(target_geom)
                .toFloat()
            )
        else:
            temp_hot_median = (
                hot_col.select("tempC")
                .median()
                .rename("tempC_hot_season_median_annual")
                .toFloat()
            )

        n_obs = col.select("ndvi").count().rename("n_obs_annual").toFloat()
        mask_valid = n_obs.gt(0).rename("mask_valid_annual").toFloat()

        out = ee.Image.cat([
            ndvi_max,
            ndvi_median,
            ndbi_median,
            temp_mean,
            temp_hot_median,
            n_obs,
            mask_valid,
        ]).clip(target_geom).toFloat()

    out = out.set({
        "year": year,
        "start_date": start.format("YYYY-MM-dd"),
        "end_date": end.advance(-1, "day").format("YYYY-MM-dd"),
        "sensors": ",".join(sensors_for_year(year)),
        "scene_count_annual": annual_count,
        "scene_count_hot_season": hot_count,
        "hot_months": "1-3,10-12",
        "system:time_start": start.millis(),
    })

    return out, annual_count, hot_count


# ============================================================
# Grade fixa de export
# ============================================================
def build_export_grid(
    target_geom: ee.Geometry,
    export_crs: str,
    pixel_size: int,
) -> tuple[list[float], ee.Geometry, int, int]:
    proj = ee.Projection(export_crs)
    bounds = target_geom.bounds(maxError=1, proj=proj)
    ring = ee.List(bounds.coordinates().get(0)).getInfo()

    xs = [pt[0] for pt in ring[:-1]]
    ys = [pt[1] for pt in ring[:-1]]

    xmin = math.floor(min(xs) / pixel_size) * pixel_size
    ymin = math.floor(min(ys) / pixel_size) * pixel_size
    xmax = math.ceil(max(xs) / pixel_size) * pixel_size
    ymax = math.ceil(max(ys) / pixel_size) * pixel_size

    width = int(round((xmax - xmin) / pixel_size))
    height = int(round((ymax - ymin) / pixel_size))

    crs_transform = [pixel_size, 0, xmin, 0, -pixel_size, ymax]
    export_region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj=proj, geodesic=False)

    print(f"Grid export: width={width}, height={height}, pixels={width * height:,}")
    print(f"Bounds snapped: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    return crs_transform, export_region, width, height


# ============================================================
# Export EE
# ============================================================
def wait_for_task(task: ee.batch.Task, poll_seconds: int) -> str:
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        print(f"Task {task.id}: {state}")

        if state == "COMPLETED":
            return state

        if state in {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}:
            error_message = status.get("error_message", "Sem detalhes.")
            raise RuntimeError(f"Task {task.id} terminou com estado {state}: {error_message}")

        time.sleep(poll_seconds)


def export_year_to_drive(
    year: int,
    img: ee.Image,
    export_region: ee.Geometry,
    export_crs: str,
    crs_transform: list[float],
    export_folder: str,
    export_prefix: str,
    file_dimensions: int,
    shard_size: int,
    max_pixels: int,
    nodata: float,
) -> tuple[ee.batch.Task, str]:
    ee.data.setWorkloadTag(f"sp-annual-{year}")

    name = f"{export_prefix}_{year}"
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=export_folder,
        fileNamePrefix=name,
        region=export_region,
        crs=export_crs,
        crsTransform=crs_transform,
        fileDimensions=file_dimensions,
        shardSize=shard_size,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
        skipEmptyTiles=True,
        formatOptions={
            "cloudOptimized": True,
            "noData": nodata,
        },
    )
    task.start()
    return task, name


def export_year_to_asset(
    year: int,
    img: ee.Image,
    export_region: ee.Geometry,
    export_crs: str,
    crs_transform: list[float],
    asset_root: str,
    export_prefix: str,
    max_pixels: int,
) -> tuple[ee.batch.Task, str]:
    ee.data.setWorkloadTag(f"sp-annual-{year}")

    name = f"{export_prefix}_{year}"
    asset_id = f"{asset_root.rstrip('/')}/{name}"

    task = ee.batch.Export.image.toAsset(
        image=img,
        description=name,
        assetId=asset_id,
        region=export_region,
        crs=export_crs,
        crsTransform=crs_transform,
        maxPixels=max_pixels,
    )
    task.start()
    return task, asset_id


def run_export(args: argparse.Namespace) -> None:
    years = parse_years(args.years)
    init_ee(args.project)

    target_geom = load_sao_paulo_ee_geometry(args.shp)
    search_geom = target_geom

    crs_transform, export_region, width, height = build_export_grid(
        target_geom=target_geom,
        export_crs=args.export_crs,
        pixel_size=args.export_scale,
    )

    print(f"FILE_DIMENSIONS={args.file_dimensions} | SHARD_SIZE={args.shard_size}")
    print(f"Pixels totais na grade: {width * height:,}")
    print(f"Modo de export: {args.export_mode}")

    for year in years:
        print(f"\nProcessando ano {year}")
        img, annual_count, hot_count = annual_composite(
            year=year,
            search_geom=search_geom,
            target_geom=target_geom,
            nodata=args.nodata,
        )
        print(f"Cenas anuais: {annual_count}")
        print(f"Cenas meses quentes: {hot_count}")
        print(f"Sensores: {sensors_for_year(year)}")

        if args.export_mode == "drive":
            task, target = export_year_to_drive(
                year=year,
                img=img,
                export_region=export_region,
                export_crs=args.export_crs,
                crs_transform=crs_transform,
                export_folder=args.export_folder,
                export_prefix=args.export_prefix,
                file_dimensions=args.file_dimensions,
                shard_size=args.shard_size,
                max_pixels=args.max_pixels,
                nodata=args.nodata,
            )
            print(f"Export para Drive iniciado: {target} | task id: {task.id}")
        else:
            if not args.asset_root:
                raise ValueError("--asset-root é obrigatório quando --export-mode asset.")
            task, target = export_year_to_asset(
                year=year,
                img=img,
                export_region=export_region,
                export_crs=args.export_crs,
                crs_transform=crs_transform,
                asset_root=args.asset_root,
                export_prefix=args.export_prefix,
                max_pixels=args.max_pixels,
            )
            print(f"Export para Asset iniciado: {target} | task id: {task.id}")

        final_state = wait_for_task(task, args.poll_seconds)
        print(f"Export finalizado: {target} | estado: {final_state}")

        if args.export_mode == "drive":
            print(
                "Aviso: os arquivos exportados permanecem no Google Drive. "
                "Este script nao baixa nem apaga arquivos do Drive automaticamente."
            )

    print("\nTodos os anos foram processados.")


# ============================================================
# Mosaico local dos tiles baixados do Drive
# ============================================================
def group_files_by_year(input_dir: Path, prefix: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    expected_prefix = f"{prefix}_"

    for tif in sorted(input_dir.glob("*.tif")):
        if not tif.name.startswith(expected_prefix):
            continue

        remainder = tif.stem[len(expected_prefix):]
        match = YEAR_PATTERN.match(remainder)
        if not match:
            continue

        year = match.group(1)
        grouped.setdefault(year, []).append(tif)

    return grouped


def mosaic_year(year: str, files: list[Path], output_dir: Path, prefix: str, nodata: float) -> Path:
    print(f"\nAno {year} | arquivos: {len(files)}")
    srcs = [rasterio.open(fp) for fp in files]

    try:
        mosaic, out_transform = merge(srcs, nodata=nodata)

        meta = srcs[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "nodata": nodata,
                "compress": "deflate",
                "predictor": 2,
                "zlevel": 6,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            }
        )

        band_descriptions = tuple(srcs[0].descriptions) if srcs and srcs[0].descriptions else tuple()

        out_path = output_dir / f"{prefix}_{year}_mosaic.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
            if band_descriptions:
                for idx, desc in enumerate(band_descriptions, start=1):
                    if desc:
                        dst.set_band_description(idx, desc)

        print(f"Mosaico salvo em: {out_path}")
        return out_path
    finally:
        for src in srcs:
            src.close()


def run_mosaic(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_files_by_year(input_dir, args.export_prefix)
    if not grouped:
        raise RuntimeError("Nenhum TIFF compatível encontrado para mosaico.")

    for year, files in grouped.items():
        mosaic_year(year, files, output_dir, args.export_prefix, args.nodata)

    print("\nTodos os mosaicos foram gerados.")


# ============================================================
# Recorte opcional por município
# ============================================================
def ensure_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("O shapefile não possui CRS definido.")
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def find_mosaic_year(path: Path) -> str:
    match = YEAR_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Não foi possível identificar o ano no nome: {path.name}")
    return match.group(1)


def run_clip(args: argparse.Namespace) -> None:
    mosaic_dir = Path(args.mosaic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf_sp = load_sao_paulo_gdf(args.shp)
    if args.cd_mun_field not in gdf_sp.columns:
        raise ValueError(
            f"Campo {args.cd_mun_field} não encontrado no shapefile. "
            f"Campos disponíveis: {gdf_sp.columns.tolist()}"
        )

    mosaics = sorted(mosaic_dir.glob("*.tif"))
    if not mosaics:
        raise RuntimeError("Nenhum mosaico .tif encontrado em --mosaic-dir.")

    for mosaic_path in mosaics:
        year = find_mosaic_year(mosaic_path)
        print(f"\nRecortando mosaico {mosaic_path.name} | ano {year}")

        with rasterio.open(mosaic_path) as src:
            gdf_year = ensure_crs(gdf_sp, src.crs)

            for idx, row in gdf_year.iterrows():
                cd_mun = str(row[args.cd_mun_field]).strip()
                geom = [row.geometry.__geo_interface__]

                try:
                    out_image, out_transform = rio_mask(
                        src,
                        geom,
                        crop=True,
                        nodata=args.nodata,
                        filled=True,
                    )
                except ValueError:
                    # sem interseção
                    continue

                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "nodata": args.nodata,
                        "compress": "deflate",
                        "predictor": 2,
                        "zlevel": 6,
                        "tiled": True,
                        "BIGTIFF": "IF_SAFER",
                    }
                )

                band_descriptions = tuple(src.descriptions) if src.descriptions else tuple()

                out_name = f"{args.export_prefix}_{year}_CD_MUN_{cd_mun}.tif"
                out_path = output_dir / out_name
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)
                    if band_descriptions:
                        for idx, desc in enumerate(band_descriptions, start=1):
                            if desc:
                                dst.set_band_description(idx, desc)

        print(f"Recortes concluídos para {mosaic_path.name}")

    print("\nTodos os recortes foram gerados.")


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline anual de Landsat para SP: export, mosaico e recorte."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export
    p_export = subparsers.add_parser("export", help="Exporta raster anual contínuo de SP via Earth Engine.")
    p_export.add_argument("--project", required=True, help="Projeto do Earth Engine / Google Cloud.")
    p_export.add_argument("--shp", required=True, help="Shapefile de municípios com a coluna CD_UF; SP sera filtrado por CD_UF == 35.")
    p_export.add_argument("--years", nargs="+", required=True, help="Anos, ex.: 2000 2010 2022")
    p_export.add_argument("--export-mode", choices=["drive", "asset"], default="drive")
    p_export.add_argument("--asset-root", default="", help="Obrigatório se export-mode=asset.")
    p_export.add_argument("--export-folder", default="gee_exports", help="Pasta no Google Drive.")
    p_export.add_argument("--export-prefix", default="landsat_annual_sp")
    p_export.add_argument("--export-scale", type=int, default=30)
    p_export.add_argument("--export-crs", default="EPSG:31983")
    p_export.add_argument("--file-dimensions", type=int, default=8192)
    p_export.add_argument("--shard-size", type=int, default=256)
    p_export.add_argument("--max-pixels", type=int, default=int(1e13))
    p_export.add_argument("--poll-seconds", type=int, default=30)
    p_export.add_argument("--nodata", type=float, default=NODATA_DEFAULT)
    p_export.set_defaults(func=run_export)

    # mosaic
    p_mosaic = subparsers.add_parser("mosaic", help="Mosaica localmente os tiles exportados do Drive.")
    p_mosaic.add_argument("--input-dir", required=True, help="Pasta com os TIFFs baixados do Drive.")
    p_mosaic.add_argument("--output-dir", required=True, help="Pasta de saída para os mosaicos anuais.")
    p_mosaic.add_argument("--export-prefix", default="landsat_annual_sp")
    p_mosaic.add_argument("--nodata", type=float, default=NODATA_DEFAULT)
    p_mosaic.set_defaults(func=run_mosaic)

    # clip
    p_clip = subparsers.add_parser("clip", help="Recorta mosaicos anuais por município.")
    p_clip.add_argument("--mosaic-dir", required=True, help="Pasta com os mosaicos anuais.")
    p_clip.add_argument("--shp", required=True, help="Shapefile de municípios com a coluna CD_UF; SP sera filtrado por CD_UF == 35.")
    p_clip.add_argument("--output-dir", required=True, help="Pasta de saída dos recortes.")
    p_clip.add_argument("--cd-mun-field", default="CD_MUN", help="Campo com o código municipal.")
    p_clip.add_argument("--export-prefix", default="landsat_annual_sp")
    p_clip.add_argument("--nodata", type=float, default=NODATA_DEFAULT)
    p_clip.set_defaults(func=run_clip)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
